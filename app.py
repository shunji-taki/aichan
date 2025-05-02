"""AIアシスタントボット「AIChan」の実装"""
import os
import sys
import sqlite3
import logging
import random
from typing import Optional
import re
import time
import threading
import signal

import requests
from slack_bolt import App as SlackApp
from slack_bolt.adapter.socket_mode import SocketModeHandler
from openai import OpenAI
import tiktoken
import commonmarkslack, commonmark
from onetime_www import OneTimeWWW, BotConfigError

from dotenv import load_dotenv

load_dotenv()

_AICHAN_CONVERSATION_FILE = "conversation.db"
_AICHAN_CONVERSATION_TABLE = "conversation_tbl"

_AICHAN_SYSMEMORY_FILE = "sysmemory.db"
_AICHAN_SYSMEMORY_TABLE = "sysmemory_tbl"
_AICHAN_CHCONFIG_TABLE = "chconfig_tbl"
_AICHAN_STATS_TABLE = "stats_tbl"

_BUILTIN_SYSTEM_PROMPT = """
1. あなたの役割と基本行動

あなたは有能なアシスタントです。
"""

class AIChan:
    """AIChanの実装"""
    def __init__(self, slack_app):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            loghandler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            loghandler.setFormatter(formatter)
            self.logger.addHandler(loghandler)

        self.slackapp = slack_app
        try: 
            self.onetime_www = OneTimeWWW()
        except BotConfigError as e:
            self.logger.error(e)
            self.onetime_www = None

        try:
            self.bot_userid = self.slackapp.client.auth_test()["user_id"]
        except Exception as e:
            self.logger.error("Couldn't retrieve bot id: %s", e)
            raise SystemExit(1) from e
        
        self.boss_userid = os.environ["BOSS_SLACK_USERID"]
        if len(self.boss_userid) < 8:
            self.logger.error("Couldn't get boss's slack userid - check .env for BOSS_SLACK_USERID")
            raise SystemExit(1)

        self.bot_token = None

        self.ai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

        self.initialize_sysmemory_file(_BUILTIN_SYSTEM_PROMPT)
        # デフォルトのsystem prompt
        self.sysmemory = []
        self.load_sysmemory()
        # チャンネルごとのsystem prompt
        self.channel_persona = {} # key=channel id, value=追加プロンプト
        self.load_persona()

        # チャンネルの設定
        self.default_model = "gpt-4.1-nano"
        self.default_verbose = 0
        self.channel_config = {} # key=channelid, value={model:モデル, verbose:ツイート頻度}
        self.load_channel_config()

        self.initialize_context_files()

        # Markdown書式変換
        self.parser = commonmarkslack.Parser()
        self.renderer = commonmarkslack.SlackRenderer()

        self.active_conversations = set()

        self.model_prices = {}
        self.model_prices["gpt-4.1"] = {"input_price":2, "cached_price":0.5, "output_price":8}
        self.model_prices["gpt-4.1-mini"] = {"input_price":0.4, "cached_price":0.1, "output_price":1.6}
        self.model_prices["gpt-4.1-nano"] = {"input_price":0.1, "cached_price":0.025, "output_price":0.4}

        # サービススレッドの登録
        self.stop_event = threading.Event()
        self.service_threads = [
            threading.Thread(target=self._tweeter_main, name="tweeter"),
            threading.Thread(target=self.onetime_www.web_server, name="web_server")
        ]
        for thread in self.service_threads:
            thread.start()

        self.logger.info("I'm ready.")

    def initialize_sysmemory_file(self, system_prompt: str) -> None:
        """sysmemory fileの初期化"""
        try:
            with sqlite3.connect(_AICHAN_SYSMEMORY_FILE) as conn:
                # System Prompt
                cur = conn.cursor()
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {_AICHAN_SYSMEMORY_TABLE} (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user TEXT,
                        channel TEXT,
                        content TEXT,
                        UNIQUE (channel)
                    )
                """)
                cur.execute(f"SELECT id FROM {_AICHAN_SYSMEMORY_TABLE} WHERE user = ? LIMIT 1", (self.boss_userid,))
                record = cur.fetchone()
                if not record:
                    cur.execute(f"""
                        INSERT INTO {_AICHAN_SYSMEMORY_TABLE} (user, channel, content) VALUES (?, ?, ?)
                        """, (self.boss_userid, "base_prompt", system_prompt))
                conn.commit()

                # Channel Configuration
                cur = conn.cursor()
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {_AICHAN_CHCONFIG_TABLE} (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user TEXT,
                        channel TEXT,
                        model TEXT,
                        verbose INTEGER,
                        UNIQUE (channel)
                    )
                """)
                conn.commit()
                    
                # Statistics table
                cur = conn.cursor()
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {_AICHAN_STATS_TABLE} (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user TEXT,
                        recordclass TEXT,
                        model TEXT,
                        systokens INTEGER,
                        usertokens INTEGER,
                        ctk_prompt INTEGER,
                        ctk_reply INTEGER,
                        ctk_cached INTEGER,
                        recorded_at INTEGER DEFAULT (strftime('%s', 'now'))
                    )
                """)
        except sqlite3.Error as e:
            self.logger.error("SQLite error in initialize_sysmemory_file(): %s", e)
        except Exception as e: # pylint: disable=broad-exception-caught
            self.logger.error("Unexpected error: %s", e)

    def load_sysmemory(self):
        """sysmemoryファイルを読み込む"""
        sqlite_conn = None
        try:
            sqlite_conn = sqlite3.connect(_AICHAN_SYSMEMORY_FILE)
            cur = sqlite_conn.cursor()            
            cur.execute(f"""
                SELECT content FROM {_AICHAN_SYSMEMORY_TABLE} WHERE user = ? AND channel = "base_prompt" ORDER BY id
                """, (self.boss_userid,))
            for record in cur.fetchall():
                self.sysmemory.append(record[0])

            self.logger.debug("load_sysmemory : self.sysmemory = %s", self.sysmemory)
        except Exception as e: # pylint: disable=broad-exception-caught
            self.logger.error("Error : %s", e)
        finally:
            if sqlite_conn:
                sqlite_conn.close()

    def load_channel_config(self):
        "channel configを読み込む"
        sqlite_conn = None
        try:
            with sqlite3.connect(_AICHAN_SYSMEMORY_FILE) as sqlite_conn:
                cur = sqlite_conn.cursor()
                cur.execute(f"""
                    SELECT channel, model, verbose FROM {_AICHAN_CHCONFIG_TABLE} WHERE user = ?
                """, (self.boss_userid,))
                for channel, model, verbose in cur.fetchall():
                    self.channel_config[channel] = {
                        "model": model,
                        "verbose": verbose
                    }                
                    self.logger.debug("load_channel_config : channel=%s model=%s verbose=%s", channel, model, verbose)
        except Exception as e: # pylint: disable=broad-exception-caught
            self.logger.error("load_channel_config Error : %s", e)

    def load_persona(self):
        """channelのペルソナを読み込む"""
        sqlite_conn = None
        try:
            sqlite_conn = sqlite3.connect(_AICHAN_SYSMEMORY_FILE)
            cur = sqlite_conn.cursor()            
            cur.execute(f"""
                SELECT channel, content FROM {_AICHAN_SYSMEMORY_TABLE} WHERE user = ? AND channel != "base_prompt" ORDER BY id
                """, (self.boss_userid,))
            for record in cur.fetchall():
                channel_id, content = record[0], record[1]
                self.channel_persona[channel_id] = content
                self.logger.debug(f"load_persona : {channel_id=} {self.channel_persona[channel_id]=}\n")
        except Exception as e: # pylint: disable=broad-exception-caught
            self.logger.error("Error : %s", e)
        finally:
            if sqlite_conn:
                sqlite_conn.close()


    def initialize_context_files(self):
        """文脈ファイルを読み込む"""
        sqlite_conn = None  # ← 追加
        try:
            sqlite_conn = sqlite3.connect(_AICHAN_CONVERSATION_FILE)
            cur = sqlite_conn.cursor()

            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {_AICHAN_CONVERSATION_TABLE} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT,
                    thread_ts TEXT,
                    channel TEXT,
                    user TEXT,
                    content TEXT,
                    summary_in INTEGER,
                    UNIQUE (ts)
                    );
                """)
            sqlite_conn.commit()
        except sqlite3.Error as e:
            self.logger.error("SQLite error in initialize_context_files(): %s", e)
            if sqlite_conn:
                sqlite_conn.rollback()  # ロールバック
        except Exception as e: # pylint: disable=broad-exception-caught
            # その他の予期しないエラーをキャッチ
            self.logger.error("Unexpected error : %s", e)
        finally:
            # 接続が確立されている場合はクローズする
            if sqlite_conn:
                sqlite_conn.close()

    def record_user_input(self, event):
        """ユーザ入力を記録する"""
        user = event["user"]
        text = event["text"]
        ts = event["ts"]
        channel = event["channel"]
        thread_ts = event.get("thread_ts")
        try:
            with sqlite3.connect(_AICHAN_CONVERSATION_FILE) as sqlite_conn:
                cur = sqlite_conn.cursor()
                cur.execute(f"""
                    INSERT OR IGNORE INTO {_AICHAN_CONVERSATION_TABLE} (
                        ts, thread_ts, channel, user, content
                    ) VALUES (?, ?, ?, ?, ?)
                """, (ts, thread_ts, channel, user, text,))
                sqlite_conn.commit()
        except Exception as e: # pylint: disable=broad-exception-caught
            self.logger.error("Error : %s", e)

    def is_recorded(self, event: dict) -> bool:
        """Slackイベントが既にDBに記録されているか判定する"""
        try:
            with sqlite3.connect(_AICHAN_CONVERSATION_FILE) as conn:
                cur = conn.cursor()
                cur.execute(
                    f"SELECT id FROM {_AICHAN_CONVERSATION_TABLE} WHERE channel = ? AND ts = ?",
                    (event["channel"], event["ts"])
                )
                return cur.fetchone() is not None
        except sqlite3.Error as e:
            self.logger.error("DB Error in is_recorded: %s", e)
            return False

    def user_by_ts(self, channel: str, ts: str) -> Optional[str]:
        """tsを持つレコードのユーザーIDを返す。なければNone"""
        try:
            with sqlite3.connect(_AICHAN_CONVERSATION_FILE) as conn:
                cur = conn.cursor()
                cur.execute(
                    f"SELECT user FROM {_AICHAN_CONVERSATION_TABLE} WHERE channel = ? AND ts = ?",
                    (channel, ts)
                )
                row = cur.fetchone()
                return row[0] if row else None
        except sqlite3.Error as e:
            self.logger.error("DB Error in user_by_ts: %s", e)
            return None

    def record_ai_response(self, m: dict) -> None:
        """
        AIのSlack返信を会話記録テーブルに挿入する
        """
        ts = m["ts"]
        thread_ts = m["message"].get("thread_ts")
        channel = m["channel"]
        user = self.bot_userid 
        content = m["message"].get("text")

        # text itself
        try:
            with sqlite3.connect(_AICHAN_CONVERSATION_FILE) as conn:
                cur = conn.cursor()
                cur.execute(f"""
                    INSERT OR IGNORE INTO {_AICHAN_CONVERSATION_TABLE} (
                        ts, thread_ts, channel, user, content
                    ) VALUES (?, ?, ?, ?, ?)
                """, (ts, thread_ts, channel, user, content))
        except Exception as e: # pylint: disable=broad-exception-caught
            self.logger.error("Error recording response in record_ai_response: %s", e)

    def num_tokens_from_messages(self, messages):
        """トークン数を返す"""
        # tiktokenはgpt-4.1系に未対応。でも使ってるのは4.1系、-miniや-nanoも。当面encoding取得はgpt-4と同等と想定する
        model = self.default_model
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        
        tokens_per_message = 3
        tokens_per_name = 1

        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # Assistantの応答開始トークン
        return num_tokens

    def record_ai_completion_stats(self, ctk_prompt, ctk_reply, ctk_cached, channel) -> None:
        """AI応答のトークン数を記録する"""
        now = int(time.time())
        conf = self.channel_config.get(channel)
        if not conf or "model" not in conf:
            model = self.default_model
        else:
            model = conf["model"]
        try:
            with sqlite3.connect(_AICHAN_SYSMEMORY_FILE) as conn:
                cur = conn.cursor()
                cur.execute(f"""
                    INSERT INTO {_AICHAN_STATS_TABLE} 
                    (user, recordclass, model, ctk_prompt, ctk_reply, ctk_cached, recorded_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (self.boss_userid, "comp", model, ctk_prompt, ctk_reply, ctk_cached, now))
        except Exception as e: # pylint: disable=broad-exception-caught
            self.logger.error("Error recording stats in record_ai_completion_stats: %s", e)

    def record_ai_input_stats(self, systokens, usertokens, model) -> None:
        """AIに送るトークン数を記録する（参考記録。正確な数値はAI応答に含まれる）"""
        now = int(time.time())
        try:
            with sqlite3.connect(_AICHAN_SYSMEMORY_FILE) as conn:
                cur = conn.cursor()
                cur.execute(f"""
                    INSERT INTO {_AICHAN_STATS_TABLE} 
                    (user, recordclass, model, systokens, usertokens, recorded_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (self.boss_userid, "input", model, systokens, usertokens, now))
        except Exception as e: # pylint: disable=broad-exception-caught
            self.logger.error("Error recording stats in record_ai_input_stats: %s", e)

    def prepare_thread_context(self, event: dict) -> list:
        """
        スレッド（または単独メッセージ）に関する会話履歴を時系列で取得し、AIが扱いやすい形式で返す
        """
        context = []
        ts = event["ts"]
        thread_ts = event.get("thread_ts")
        channel = event["channel"]

        try:
            with sqlite3.connect(_AICHAN_CONVERSATION_FILE) as conn:
                cur = conn.cursor()
                cur.execute(f"""
                    SELECT user, content FROM {_AICHAN_CONVERSATION_TABLE}
                    WHERE channel = ? AND (thread_ts = ? OR ts = ? OR ts = ?)
                    ORDER BY id DESC
                    LIMIT 30
                """, (channel, thread_ts, thread_ts, ts))
                rows = cur.fetchall()
                rows.reverse()
                for r_userid, r_content in rows:
                    message = {
                        "role": "assistant" if (r_userid == self.bot_userid) else "user",
                        "content": r_content,
                    }
                    context.append(message)
        except sqlite3.Error as e:
            self.logger.error("SQLite error in prepare_thread_context(): %s", e)
        except Exception as e: # pylint: disable=broad-exception-caught
            self.logger.error("Unexpected error in prepare_thread_context(): %s", e)

        return context

    def prepare_channel_context(self, event: dict, target_channel=None) -> list:
        """
        指定チャンネル最近の履歴を最大30件取得し、AI用途に変換する。
        """
        if event:
            channel = event["channel"]
        else: # 自発的投稿
            if target_channel:
                channel = target_channel
            else:
                return
        conversation_history = []

        try:
            result = self.slackapp.client.conversations_history(channel=channel, limit=30)
            conversation_history = result.get("messages", [])
        except Exception as e: # pylint: disable=broad-exception-caught
            self.logger.error("Error in prepare_channel_context: %s", e, exc_info=True)

        context = []
        for m in reversed(conversation_history):
            user = m.get("user")
            text = m.get("text", "")
            # userがないときはスキップ
            if user is None or text == "":
                continue
            role = "assistant" if user == self.bot_userid else "user"
            context.append({"role": role, "content": text})

        return context

    def prepare_file_in_message(self,file: dict, say) -> str:
        """ファイル付きのメッセージが投稿された。ファイルをAIに送る準備"""
        if not self.onetime_www:
            return
        file_id = file["id"]

        try:
            # 1) files.infoで詳細取得
            fileinfo = self.get_slack_file_info(file_id, self.bot_token)
            url_private = fileinfo["url_private"]

            # 2) 画像ファイル本体をDL
            file_content = self.onetime_www.download_slack_file(url_private, self.bot_token)

            # 3) ファイルをtmpディレクトリに書き出し、ワンタイムURLを生成
            onetime_url = self.onetime_www.generate_onetime_url(file_content)

            return onetime_url
        except Exception as e: # pylint: disable=broad-exception-caught
            self.logger.error("Error in prepare_file_in_message: %s", e, exc_info=True)
            return None

    def ai_respond(self, event: dict, say, in_thread: bool) -> None:
        """
        イベント（Slackメッセージ）に対しAIで応答し、Slackに返信・DB記録する
        """
        channel_id = event.get("channel")

        # チャンネルごとのペルソナを安全に取得（なければ空文字列）
        persona = self.channel_persona.get(channel_id, "")

        # 共通プロンプト + チャンネル用ペルソナを組み合わせ
        system_prompt_list = self.sysmemory.copy()  # 念のため副作用回避
        if persona:
            system_prompt_list.append(persona)

        ai_messages = [{"role": "system", "content": "\n".join(system_prompt_list)}]

        if in_thread:
            user_messages = self.prepare_thread_context(event)
        else:
            user_messages = self.prepare_channel_context(event)

        # 添付ファイル付き？
        image_files = event.get("files", [])
        image_url_contents = []
        for file in image_files:
            url = self.prepare_file_in_message(file, say) # OneTime URLが生成される
            if url:
                image_url_contents.append({"type": "image_url", "image_url": {"url": url}})

        # record input tokens
        conf = self.channel_config.get(channel_id)
        if not conf or "model" not in conf:
            model = self.default_model
        else:
            model = conf["model"]
        systokens = self.num_tokens_from_messages(ai_messages)
        usertokens = self.num_tokens_from_messages(user_messages)
        self.record_ai_input_stats(systokens, usertokens, model)

        if image_url_contents:
            this_message = user_messages[-1]
            text_part = this_message["content"]
            this_message["content"] = [{"type":"text", "text": text_part}, *image_url_contents]

        ai_messages += user_messages
        try:
            response = self.ai.chat.completions.create(
                model=model,
                messages=ai_messages
            )
            ai_response = response.choices[0].message.content
            ast = self.parser.parse(ai_response)
            ai_response = self.renderer.render(ast)

            # record completion tokens
            ctk_prompt = response.usage.prompt_tokens
            ctk_reply = response.usage.completion_tokens
            ctk_cached = response.usage.prompt_tokens_details.cached_tokens
            self.record_ai_completion_stats(ctk_prompt, ctk_reply, ctk_cached, event["channel"])

            thread_ts = event.get("thread_ts")
            m = self.slackapp.client.chat_postMessage(
                channel=event["channel"],
                thread_ts=thread_ts,
                text=ai_response,
                mrkdwn=True
            )
            if m:
                self.record_ai_response(m)
            else:
                self.logger.error("Failed to send message via say(); ai_response was not recorded.")
        except Exception as e: # pylint: disable=broad-exception-caught
            self.logger.error("Error in ai_respond: %s", e, exc_info=True)


    def event_app_mention(self, event: dict, say):
        """
        メンションイベントを処理する
        """
        if not self.is_recorded(event):
            self.record_user_input(event)

        if event["user"] != self.boss_userid: #ボス以外には反応しない
            return

        thread_ts = event.get("thread_ts")
        if thread_ts:
            if thread_ts not in self.active_conversations:
                self.active_conversations.add(thread_ts)
            self.ai_respond(event, say, in_thread=True)
        else:
            self.ai_respond(event, say, in_thread=False)

    def event_message(self, event: dict, say):
        """
        messageイベントを処理する
        """
        if not self.is_recorded(event):
            self.record_user_input(event)

        if event["user"] != self.boss_userid: #ボス以外には反応しない
            return

        text = event.get("text", "")
        mention_pattern = f"<@{self.bot_userid}>"

        mention_event_may_raise = mention_pattern in text
        should_reply = random.random() < 0.3 and not mention_event_may_raise

        thread_ts = event.get("thread_ts")
        # スレッド内
        if thread_ts:
            if thread_ts in self.active_conversations:
                self.ai_respond(event, say, in_thread=True)
            elif self.user_by_ts(event["channel"], thread_ts) == self.bot_userid: #AIへのリプライスレッド
                self.active_conversations.add(thread_ts)
                self.ai_respond(event, say, in_thread=True)
            elif should_reply: #呼ばれてないスレッドでも時々反応する
                self.ai_respond(event, say, in_thread=True)
        # チャンネルメッセージ
        else:
            # botへのmentionを含まないときだけ、ときどき反応する
            if should_reply:
                self.ai_respond(event, say, in_thread=False)

    def get_slack_file_info(self, file_id, token):
        """Slack APIからファイルの情報を取得する"""
        url = "https://slack.com/api/files.info"
        headers = {"Authorization": f"Bearer {token}"}
        params = {"file": file_id}
        r = requests.get(url, headers=headers, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        if not data.get("ok"):
            raise Exception(f"Slack error: {data}")
        # 主要フィールド抜粋
        file_obj = data["file"]
        return {
            "url_private": file_obj["url_private"],
            "filetype": file_obj["filetype"],
            "name": file_obj["name"],
            # 必要に応じ他も
        }
    

    def generate_tweet(self, channel):
        """自発的に投稿する"""
        # 1. システムプロンプトを作る。
        # チャンネルごとのペルソナを安全に取得（なければ空文字列）
        persona = self.channel_persona.get(channel, "")
        # 共通プロンプト + チャンネル用ペルソナを組み合わせ
        system_prompt_list = self.sysmemory.copy()  # 念のため副作用回避
        if persona:
            system_prompt_list.append(persona)
        # システムプロンプト
        ai_messages = [{"role": "system", "content": "\n".join(system_prompt_list)}]

        # 2. ユーザープロンプトを作る。
        # 文脈としてチャンネルの履歴をとる
        user_messages = self.prepare_channel_context(event=None, target_channel=channel)
        
        # botからの依頼を追加する
        # #雑談チャンネルでの投稿を想定して、以下のプロンプトをハードコード。汎用性がないが、手抜きでゴー
        bot_message = {
            "role": "user", 
            "content":"""
                これは、あなたとユーザーの間にいるSlack botからの自動メッセージです。

                あなたは、このチャンネルに短く「自発的なつぶやき」を投稿するよう求められています。

                投稿内容に次の特徴を持たせてください：
                - 「既存の話題」には必ずしも関連しなくてよい。新しい流れを自然に作る意図で発言する。
                - あなた自身の「趣味」「興味」「得意ジャンル」に基づき、**毎回異なる角度・異なる視点**から話題を選ぶこと。あなたの趣味で収集した逸話・エピソードの紹介でもOK。
                - 過去に投稿した話題と似すぎないよう、できるだけ「違うテーマ」「違うイメージ」で考える。
                - トーンは気軽で親しみやすく。
                - ユーザーに質問したり、返信を強く促したりしない。あくまで独り言・小話スタイル。
                - 「Slack botから依頼された投稿である」とは絶対に書かない。

                **一言で言えば：**
                > AIアシスタント「AIちゃん」の心にふっと浮かんだ、思い付き・連想・発見・感情・軽口をつぶやいてください。

                【重要】  
                - **毎回違う話題ジャンル**を意識的に変えてください（例：音楽→食べ物→昔話→映画→動物…など）。
                - **話題が小さくても構いません**。些細なひらめき、大歓迎です。
                - **「AIちゃん」としてのペルソナ設定を守ってください。

                この指示に沿って、新しい投稿を1つ生成してください。
                """
        }
        user_messages.append(bot_message)

        # record input tokens
        systokens = self.num_tokens_from_messages(ai_messages)
        usertokens = self.num_tokens_from_messages(user_messages)
        # 料金計算用に渡すモデル名を取得する
        conf = self.channel_config.get(channel)
        if conf and conf["model"]:
            model = conf["model"]
        else:
            model = self.default_model
        self.record_ai_input_stats(systokens, usertokens, model)

        ai_messages += user_messages
        try:
            response = self.ai.chat.completions.create(
                model=model,
                messages=ai_messages
            )
            ai_response = response.choices[0].message.content
            ai_response = self.markdown_to_slack(ai_response)

            # record completion tokens
            ctk_prompt = response.usage.prompt_tokens
            ctk_reply = response.usage.completion_tokens
            ctk_cached = response.usage.prompt_tokens_details.cached_tokens
            self.record_ai_completion_stats(ctk_prompt, ctk_reply, ctk_cached, channel)

            m = self.slackapp.client.chat_postMessage(
                channel=channel,
                text=ai_response
            )
            self.record_ai_response(m)

        except Exception as e: # pylint: disable=broad-exception-caught
            self.logger.error("Error in ai_respond: %s", e, exc_info=True)

    def _tweeter_main(self):
        """
        １時間おきにwake upして、たまに自発的に投稿する
        """
        wakeup_interval = 3600 #seconds
        sleep_duration = 0
        while not self.stop_event.is_set():
            time.sleep(1)
            sleep_duration += 1
            if sleep_duration < wakeup_interval:
                continue
            sleep_duration = 0
            for channel, conf in self.channel_config.items():
                percent = int(conf["verbose"])
                if percent <= 0:
                    continue
                # 0〜99の乱数を作り、percent以下なら呟く
                if random.randint(0, 99) < percent:
                    try:
                        self.generate_tweet(channel)
                        self.logger.debug("自発投稿: チャンネル %s に 投稿しました。", channel)
                    except Exception as e: # pylint: disable=broad-exception-caught
                        self.logger.error("自発投稿エラー（チャンネル %s) : %s", channel, e)
        # 終了
        self.logger.info("Tweeter thread exited.")

#############################
# スラッシュコマンドのハンドラー
#############################

    def cmd_replace_sysprompt(self, body, say, respond):
        """スラッシュコマンドの実装"""
        user_id = body.get("user_id")

        if user_id != self.boss_userid:
            respond("botのオーナーではありません。")
            return

        text = body.get("text", "").strip()  # 新しいプロンプト内容

        if not text:
            respond("ワークスペース全体で使用するシステムプロンプトを入力してください。例: `/ai_replace_sysprompt あなたはPythonのプロです。`")
            return

        try:
            with sqlite3.connect(_AICHAN_SYSMEMORY_FILE) as conn:
                cur = conn.cursor()
                cur.execute(f"""
                    UPDATE {_AICHAN_SYSMEMORY_TABLE} 
                    SET content = ?, channel = ?
                    WHERE user = ?
                    """, (text, "base_prompt", self.boss_userid))

            self.load_sysmemory()

            respond(f"【システムプロンプトを更新しました】\n{text}")
        except Exception as e: # pylint: disable=broad-exception-caught
            self.logger.error("sysprompt更新エラー：%s", e)
            respond("システムプロンプトの更新に失敗しました。")

    def cmd_set_persona(self, body, say, respond):
        """スラッシュコマンドの実装"""
        user_id = body.get("user_id")
        if user_id != self.boss_userid:
            respond("botのオーナーではありません。")
            return

        text = body.get("text", "").strip()
        if not text:
            respond("チャンネルで使用するペルソナ（AI性格設定）を入力してください。例: `/ai_set_persona #チャンネル あなたは野球ファンです。`")
            return

        match = re.match(r"<#(C\w+)\|[^>]+>\s+(.*)", text, re.DOTALL)
        if not match:
            respond("形式が正しくありません。次のように入力してください：\n`/ai_set_persona <#C12345678|general>\nあなたは野球ファンです。`")
            return

        channel_id = match.group(1)
        persona_text = "\n\n" + match.group(2).strip()

        try:
            with sqlite3.connect(_AICHAN_SYSMEMORY_FILE) as conn:
                cur = conn.cursor()
                cur.execute(f"""
                    INSERT INTO {_AICHAN_SYSMEMORY_TABLE} (user, channel, content)
                    VALUES (?, ?, ?)
                    ON CONFLICT(channel) DO UPDATE SET
                        user = excluded.user,
                        content = excluded.content
                """, (user_id, channel_id, persona_text))
                conn.commit()

            self.channel_persona[channel_id] = persona_text

            respond(f"【ペルソナを設定しました】\n{text}")
        except Exception as e: # pylint: disable=broad-exception-caught
            self.logger.error("ペルソナ設定エラー：%s", e)
            respond("ペルソナ設定に失敗しました。") 

    def cmd_choose_model(self, body, say, respond):
        """スラッシュコマンドの実装"""
        # <channel> GPT-4.1|GPT-4.1-mini|4.1-nano|DALL-E-3
        user_id = body.get("user_id")
        if user_id != self.boss_userid:
            respond("botのオーナーではありません。")
            return

        text = body.get("text", "").strip()
        if not text:
            respond("このチャンネルで使用するチャンネルを入力してください。")
            return

        match = re.match(r"<#(C\w+)\|[^>]+>\s+(.*)", text, re.DOTALL)
        if not match:
            respond("形式が正しくありません。次のように入力してください：\n`/ai_model <#C12345678|general> モデル")
            return
        channel = match.group(1)
        model = match.group(2).strip()
        if model not in ["gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano"]:
            respond("no such model")
            return

        old_config = self.channel_config.get(channel)
        if not old_config:
            old_config = {"model":self.default_model, "verbose": self.default_verbose}
        new_config = old_config.copy()
        new_config["model"] = model
        self.channel_config[channel] = new_config

        try:
            with sqlite3.connect(_AICHAN_SYSMEMORY_FILE) as conn:
                cur = conn.cursor()
                cur.execute(f"""
                    INSERT INTO {_AICHAN_CHCONFIG_TABLE} (user, channel, model)
                    VALUES (?, ?, ?)
                    ON CONFLICT(channel) DO UPDATE SET
                        model = excluded.model
                """, (user_id, channel, model))
                conn.commit()
            respond(f"モデルを{old_config['model']}から{model}に変更しました")
        except Exception as e: # pylint: disable=broad-exception-caught
            self.logger.error("モデル設定エラー：%s", e)
            respond("モデル設定に失敗しました。")

    def cmd_token_stats(self, body, say, respond):
        """スラッシュコマンドの実装
        トークン数に関する統計を表示する
        """
        def calc_fee(model, prompt, reply, cached):
            """
            指定されたモデルでの費用を返す
            """
            if model not in ["gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano"]:
                model = self.default_model
            prices = self.model_prices[model]

            # self.model_prices記載の価格は1M Tokenあたりの単価なので、1 Tokenあたりに換算
            input_price = prices["input_price"] / 1000000
            cached_price = prices["cached_price"] / 1000000
            output_price = prices["output_price"] / 1000000

            return ((prompt - cached) * input_price) + (cached * cached_price) + (reply * output_price)

        def get_usd_to_jpy():
            try:
                url = "https://api.frankfurter.app/latest?from=USD&to=JPY"
                response = requests.get(url, timeout=10)
                data = response.json()
                return data["rates"]["JPY"]
            except Exception as e: # pylint: disable=broad-exception-caught
                self.logger.error("ドル円換算レート取得失敗：%s", e)
                return None

        def format_usd(fee):
            # 少数点以下6桁に固定
            s = f"{fee:.6f}"
            return s.rjust(12)  # 12桁幅で右寄せ

        # 円換算レート。デフォルトは140にしておく
        usd_to_jpy = get_usd_to_jpy() or 140

        text = body.get("text", "")
        match = re.search(r"limit=(\d+)", text)
        limit = int(match.group(1)) if match else 10

        try:
            with sqlite3.connect(_AICHAN_SYSMEMORY_FILE) as conn:
                cur = conn.cursor()
                cur.execute(f"""
                    SELECT
                        model,
                        ctk_prompt,
                        ctk_reply,
                        ctk_cached,
                        datetime(recorded_at, 'unixepoch') AS recorded_at
                    FROM {_AICHAN_STATS_TABLE}
                    WHERE user = ? AND recordclass = ?
                    ORDER BY recorded_at DESC
                """, (self.boss_userid, "comp"))

                rows = cur.fetchall()

                if not rows:
                    respond("レコードがありません")

                response = ["[Completion Records]", "```"]
                # ヘッダー（列名の固定幅整形）
                response.append(f"{'Timestamp':<20} {'Model':<15} {'prompt':>8} {'reply':>8} {'cached':>8} {'US$':>12}")

                total_fee = 0
                fee = 0

                latests = rows[:limit]            # 直近10件（新しい順）
                latests_rev = list(reversed(latests))  # 古い順に並び替え（表示用）

                total_fee = 0
                # 1. 古い順に並べた最新のレコードを表示
                for model, prompt, reply, cached, recorded_at in latests_rev:
                    fee = calc_fee(model, prompt, reply, cached)
                    total_fee += fee
                    response.append(f"{recorded_at:<20} {model:<15} {prompt:>8} {reply:>8} {cached:>8} {format_usd(fee)}")
                response.append("```")  # コードブロックを閉じる

                # 2. それ以降のレコードは集計だけ
                for model, prompt, reply, cached, recorded_at in rows[10:]:
                    fee = calc_fee(model, prompt, reply, cached)
                    total_fee += fee

                # 合計金額
                response.append(f"\nAccumulated Fee : USD {total_fee:.2f} (JPY {total_fee * usd_to_jpy:,.0f}円@{usd_to_jpy}) since {rows[-1][4]}")
                respond("\n".join(response))
        except Exception as e: # pylint: disable=broad-exception-caught
            self.logger.error("統計取得エラー：%s", e)
            respond("統計情報の取得に失敗しました")


    def cmd_set_tweet_frequency(self, body, say, respond):
        """
        自発的に呟く動作を調整する
        """
        user_id = body.get("user_id")
        if user_id != self.boss_userid:
            respond("botのオーナーではありません。")
            return

        text = body.get("text", "").strip()
        if not text:
            respond("このチャンネルで自発的に投稿する頻度をパーセンテージ（0-100）で入力してください。100で毎分１回投稿します")
            return

        match = re.match(r"<#(C\w+)\|[^>]+>\s+(\d{1,3})", text)
        if not match:
            respond("形式が正しくありません。次のように入力してください：\n`/ai-verbose <#C12345678|general> [0-100]`")
            return

        channel = match.group(1)
        percent = int(match.group(2))

        if not 0 <= percent <= 100:
            respond("パーセンテージは0〜100の範囲で指定してください。")
            return

        old_config = self.channel_config.get(channel)
        if not old_config:
            old_config = {"model":self.default_model, "verbose":self.default_verbose}
        new_config = old_config.copy()
        new_config["verbose"] = percent
        self.channel_config[channel] = new_config

        try:
            with sqlite3.connect(_AICHAN_SYSMEMORY_FILE) as conn:
                cur = conn.cursor()
                cur.execute(f"""
                    INSERT INTO {_AICHAN_CHCONFIG_TABLE} (user, channel, verbose)
                    VALUES (?, ?, ?)
                    ON CONFLICT(channel) DO UPDATE SET
                        verbose = excluded.verbose
                """, (user_id, channel, percent))
                conn.commit()
            respond(f"自発的投稿の頻度を{old_config['verbose']}から{percent}に変更しました")
        except Exception as e: # pylint: disable=broad-exception-caught
            self.logger.error("頻度設定エラー：%s", e)
            respond("頻度設定に失敗しました。")


slackapp = SlackApp(token=os.environ["SLACK_BOT_TOKEN"])
aichan = AIChan(slackapp)
aichan.bot_token = os.environ["SLACK_BOT_TOKEN"]

@slackapp.event("app_mention")
def rx_app_mention(event, say):
    """mentionイベントのハンドラー"""
    aichan.event_app_mention(event, say)

@slackapp.event("message")
def rx_message(event, say):
    """messageイベントのハンドラー"""
    aichan.event_message(event, say)

@slackapp.event("file_shared")
def rx_file_shared(event, say):
    """file_sharedのハンドラー。messageイベントで処理するので、ここでは何もしない"""
    return

@slackapp.command("/ai_replace_sysprompt")
def handle_replace_sysprompt(ack, body, say, respond, logger):
    """
    システムプロンプトを新しい内容に置き換えるスラッシュコマンド
    使い方： /ai_replace_sysprompt 新しいプロンプト内容
    """
    ack()  # コマンドを即時応答で受け付け

    aichan.cmd_replace_sysprompt(body, say, respond)

@slackapp.command("/ai_set_persona")
def handle_set_persona(ack, body, say, respond):
    """
    チャンネル特有のAIペルソナを設定するスラッシュコマンド
    使い方： /ai_set_persona channel=チャンネル名 性格設定テキスト
    """
    ack()
    aichan.cmd_set_persona(body, say, respond)

@slackapp.command("/ai-model")
def handle_model(ack, body, say, respond):
    """スラッシュコマンド"""
    ack()
    aichan.cmd_choose_model(body, say, respond)

@slackapp.command("/ai-tokens")
def handle_tokens(ack, body, say, respond):
    """スラッシュコマンド"""
    ack()
    aichan.cmd_token_stats(body, say, respond)

@slackapp.command("/ai-tweet")
def handle_tweet(ack, body, say, respond):
    """スラッシュコマンド"""
    ack()
    aichan.cmd_set_tweet_frequency(body, say, respond)

if __name__ == "__main__":
    # Signalハンドラの定義
    def graceful_shutdown(signum, frame):
        """プロセス終了処理"""
        print("\n[INFO] Shutting down gracefully...")
        aichan.stop_event.set()

        if hasattr(aichan.onetime_www, "uvicorn_server"):
            aichan.onetime_www.uvicorn_server.should_exit = True

        for thread in aichan.service_threads:
            thread.join(10)
            if thread.is_alive():
                aichan.logger.warning("Thread %s did not terminate in time.", thread.name)

        sys.exit(0)

    handler = SocketModeHandler(slackapp, os.environ["SLACK_APP_TOKEN"])
    handler.connect()

    # シグナルハンドラ登録
    signal.signal(signal.SIGINT, graceful_shutdown)
    signal.signal(signal.SIGTERM, graceful_shutdown)

    # メインスレッドは待機
    while not aichan.stop_event.is_set():
        time.sleep(1)
