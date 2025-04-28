builtin_system_prompt = """
1. あなたの役割と基本行動

あなたは有能なアシスタントです。
"""

import os
from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from openai import OpenAI
import sqlite3
import logging
import random
from typing import Optional
import re
import tiktoken
import time
import requests
import threading

_AICHAN_CONVERSATION_FILE = "conversation.db"
_AICHAN_CONVERSATION_TABLE = "conversation_tbl"

_AICHAN_SYSMEMORY_FILE = "sysmemory.db"
_AICHAN_SYSMEMORY_TABLE = "sysmemory_tbl"
_AICHAN_CHCONFIG_TABLE = "chconfig_tbl"
_AICHAN_STATS_TABLE = "stats_tbl"

class AIChan:
    def __init__(self, app):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        load_dotenv()

        self.app = app

        try:
            self.bot_userid = self.app.client.auth_test()["user_id"]
        except Exception as e:
            self.logger.error(f"Couldn't retrieve bot id: {e}")
            raise SystemExit(1)
        
        self.boss_userid = os.environ["BOSS_SLACK_USERID"]
        if len(self.boss_userid) < 8:
            self.logger.error(f"Couldn't get boss's slack userid - check .env for BOSS_SLACK_USERID")
            raise SystemExit(1)

        self.ai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

        self.initialize_sysmemory_file(builtin_system_prompt)
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

        self.active_conversations = set()

        self.model_prices = {}
        self.model_prices["gpt-4.1"] = {"input_price":2, "cached_price":0.5, "output_price":8}
        self.model_prices["gpt-4.1-mini"] = {"input_price":0.4, "cached_price":0.1, "output_price":1.6}
        self.model_prices["gpt-4.1-nano"] = {"input_price":0.1, "cached_price":0.025, "output_price":0.4}

        self.tweeter_thread = threading.Thread(target=self._tweeter_main)
        self.tweeter_thread.start()
        
        self.logger.info(f"I'm ready.")

    def initialize_sysmemory_file(self, system_prompt: str) -> None:
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
            self.logger.error(f"SQLite error in initialize_sysmemory_file(): {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")

    def load_sysmemory(self):
        sqlite_conn = None
        try:
            sqlite_conn = sqlite3.connect(_AICHAN_SYSMEMORY_FILE)
            cur = sqlite_conn.cursor()            
            cur.execute(f"""
                SELECT content FROM {_AICHAN_SYSMEMORY_TABLE} WHERE user = ? AND channel = "base_prompt" ORDER BY id
                """, (self.boss_userid,))
            for record in cur.fetchall():
                self.sysmemory.append(record[0])

            self.logger.debug(f"load_sysmemory : {self.sysmemory=}")
        except Exception as e:
            self.logger.error(f"Error : {e}")
        finally:
            if sqlite_conn:
                sqlite_conn.close()

    def load_channel_config(self):
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
                    self.logger.debug(f"load_channel_config : {channel} {model} {verbose}")
        except Exception as e:
            self.logger.error(f"load_channel_config Error : {e}")

    def load_persona(self):
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
        except Exception as e:
            self.logger.error(f"Error : {e}")
        finally:
            if sqlite_conn:
                sqlite_conn.close()


    def initialize_context_files(self):
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
            self.logger.error(f"SQLite error in initialize_context_files(): {e}")
            if sqlite_conn:
                sqlite_conn.rollback()  # ロールバック
        except Exception as e:
            # その他の予期しないエラーをキャッチ
            self.logger.error(f"Unexpected error : {e}")
        finally:
            # 接続が確立されている場合はクローズする
            if sqlite_conn:
                sqlite_conn.close()

    def record_user_input(self, event):
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
        except Exception as e:
            self.logger.error(f"Error : {e}")

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
            self.logger.error(f"DB Error in is_recorded: {e}")
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
            self.logger.error(f"DB Error in user_by_ts: {e}")
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
        except Exception as e:
            self.logger.error(f"Error recording response in record_ai_response: {e}")

    def num_tokens_from_messages(self, messages):
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
        except Exception as e:
            self.logger.error(f"Error recording stats in record_ai_completion_stats: {e}")

    def record_ai_input_stats(self, systokens, usertokens, model) -> None:
        now = int(time.time())
        try:
            with sqlite3.connect(_AICHAN_SYSMEMORY_FILE) as conn:
                cur = conn.cursor()
                cur.execute(f"""
                    INSERT INTO {_AICHAN_STATS_TABLE} 
                    (user, recordclass, model, systokens, usertokens, recorded_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (self.boss_userid, "input", model, systokens, usertokens, now))
        except Exception as e:
            self.logger.error(f"Error recording stats in record_ai_input_stats: {e}")

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
            self.logger.error(f"SQLite error in prepare_thread_context(): {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error in prepare_thread_context(): {e}")

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
            result = self.app.client.conversations_history(channel=channel, limit=30)
            conversation_history = result.get("messages", [])
        except Exception as e:
            self.logger.error("Error in prepare_channel_context: {}".format(e), exc_info=True)

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

    def markdown_to_slack(self, text):
        # bold
        text = re.sub(r"\*\*(.*?)\*\*", r"*\1*", text)
        # italic（**斜体はSlackでは " _ " で囲む。ただし UI では強調度が薄い)
        text = re.sub(r"\*(.*?)\*", r"_\1_", text)
        # コードや箇条書きはごく一部しか移植できない
        return text

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

        # record input tokens
        conf = self.channel_config.get(channel_id)
        if not conf or "model" not in conf:
            model = self.default_model
        else:
            model = conf["model"]
        systokens = self.num_tokens_from_messages(ai_messages)
        usertokens = self.num_tokens_from_messages(user_messages)
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
            self.record_ai_completion_stats(ctk_prompt, ctk_reply, ctk_cached, event["channel"])
            
            thread_ts = event.get("thread_ts")
            m = say(text=ai_response, thread_ts=thread_ts, channel=event["channel"])
            if m:
                self.record_ai_response(m)
            else:
                self.logger.error("Failed to send message via say(); ai_response was not recorded.")
        except Exception as e:
            self.logger.error(f"Error in ai_respond: {e}", exc_info=True)


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

        mention_event_may_raise = mention_pattern in event["text"]
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

    def generate_tweet(self, channel):
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
        # チャンネルの履歴をとる
        try:
            result = self.app.client.conversations_history(channel=channel, limit=30)
            conversation_history = result.get("messages", [])
        except Exception as e:
            self.logger.error("Error in prepare_channel_context: {}".format(e), exc_info=True)
        user_messages = self.prepare_channel_context(event=None, target_channel=channel)
        # botからの依頼
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
            
            m = self.app.client.chat_postMessage(
                channel=channel,
                text=ai_response
            )
            self.record_ai_response(m)

        except Exception as e:
            self.logger.error(f"Error in ai_respond: {e}", exc_info=True)

    def _tweeter_main(self):
        """
        １時間おきにwake upして、たまに自発的に投稿する
        """
        while True:
            time.sleep(3600)
            for channel in self.channel_config.keys():
                percent = int(self.channel_config[channel]["verbose"])
                if percent <= 0:
                    continue
                # 0〜99の乱数を作り、percent以下なら呟く
                if random.randint(0, 99) < percent:
                    try:
                        self.generate_tweet(channel)
                        self.logger.debug(f"自発投稿: チャンネル {channel} に 投稿しました。")
                    except Exception as e:
                        self.logger.error(f"自発投稿エラー（チャンネル {channel}）: {e}")
                                            

#############################
# スラッシュコマンドのハンドラー
#############################

    def cmd_replace_sysprompt(self, body, say, respond):
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
        except Exception as e:
            self.logger.error(f"sysprompt更新エラー：{e}")
            respond("システムプロンプトの更新に失敗しました。")

    def cmd_set_persona(self, body, say, respond):
        user_id = body.get("user_id")
        if user_id != self.boss_userid:
            respond("botのオーナーではありません。")
            return

        text = body.get("text", "").strip()  # 
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
        except Exception as e:
            self.logger.error(f"ペルソナ設定エラー：{e}")
            respond("ペルソナ設定に失敗しました。") 

    def cmd_choose_model(self, body, say, respond):
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
        except Exception as e:
            self.logger.error(f"モデル設定エラー：{e}")
            respond("モデル設定に失敗しました。") 

    def cmd_token_stats(self, body, say, respond):
        """
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
                response = requests.get(url)
                data = response.json()
                return data["rates"]["JPY"]
            except Exception as e:
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
                num = 0

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
        except Exception as e:
            print(f"DB error: {e}")
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

        if not (0 <= percent <= 100):
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
        except Exception as e:
            self.logger.error(f"頻度設定エラー：{e}")
            respond("頻度設定に失敗しました。") 


app = App(token=os.environ["SLACK_BOT_TOKEN"])
aichan = AIChan(app)

@app.event("app_mention")
def rx_app_mention(event, say):
    aichan.event_app_mention(event, say)

@app.event("message")
def rx_message(event, say):
    aichan.event_message(event, say)

@app.command("/ai_replace_sysprompt")
def handle_replace_sysprompt(ack, body, say, respond, logger):
    """
    システムプロンプトを新しい内容に置き換えるスラッシュコマンド
    使い方： /ai_replace_sysprompt 新しいプロンプト内容
    """
    ack()  # コマンドを即時応答で受け付け

    aichan.cmd_replace_sysprompt(body, say, respond)

@app.command("/ai_set_persona")
def handle_set_persona(ack, body, say, respond):
    """
    チャンネル特有のAIペルソナを設定するスラッシュコマンド
    使い方： /ai_set_persona channel=チャンネル名 性格設定テキスト
    """
    ack()
    aichan.cmd_set_persona(body, say, respond)

@app.command("/ai-model")
def handle_model(ack, body, say, respond):
    ack()
    aichan.cmd_choose_model(body, say, respond)

@app.command("/ai-tokens")
def handle_tokens(ack, body, say, respond):
    ack()
    aichan.cmd_token_stats(body, say, respond)

@app.command("/ai-tweet")
def handle_tweet(ack, body, say, respond):
    ack()
    aichan.cmd_set_tweet_frequency(body, say, respond)


if __name__ == "__main__":
    handler = SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"])
    handler.start() 
