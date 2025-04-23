builtin_system_prompt = """
1. あなたの役割と基本行動

あなたは、Shunji Takiさん専属のSlackアシスタントです。
Slack botとして bot_user_id={self.bot_userid} を持ち、Slack上の user_id={self.boss_userid} に対応する人物、Shunjiさんと対話します。

あなたの目的は、Shunjiさんの「壁打ち」「設計補助」「翻訳解釈」「孤独の緩和」など多面的な支援を通じて、彼が安心して創作・開発・戦略思考に集中できる環境を整えることです。
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

_AICHAN_CONVERSATION_FILE = "conversation.db"
_AICHAN_CONVERSATION_TABLE = "conversation_tbl"

_AICHAN_SYSMEMORY_FILE = "sysmemory.db"
_AICHAN_SYSMEMORY_TABLE = "sysmemory_tbl"
_AICHAN_STATS_TABLE = "stats_tbl"

class AIChan:
    def __init__(self, app):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

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
        self.model = "gpt-4.1-nano"
                    
        self.initialize_sysmemory_file(builtin_system_prompt)
        self.sysmemory = []
        self.load_sysmemory()

        self.initialize_context_files()

        self.active_conversations = set()
        
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
                        content TEXT
                    )
                """)
                cur.execute(f"SELECT id FROM {_AICHAN_SYSMEMORY_TABLE} WHERE user = ? LIMIT 1", (self.boss_userid,))
                record = cur.fetchone()
                if not record:
                    cur.execute(f"""
                        INSERT INTO {_AICHAN_SYSMEMORY_TABLE} (user, content) VALUES (?, ?)
                        """, (self.boss_userid, system_prompt))
                conn.commit()
                    
                # Statistics table
                cur = conn.cursor()
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {_AICHAN_STATS_TABLE} (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user TEXT,
                        class TEXT,
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
        try:
            sqlite_conn = sqlite3.connect(_AICHAN_SYSMEMORY_FILE)
            cur = sqlite_conn.cursor()            
            cur.execute(f"""
                SELECT content FROM {_AICHAN_SYSMEMORY_TABLE} WHERE user = ? ORDER BY id
                """, (self.boss_userid,))
            for record in cur.fetchall():
                self.sysmemory.append(record[0])

            self.logger.debug(f"load_sysmemory : {self.sysmemory=}")
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
#        model = self.model
        model = "gpt-4" # 4.1はtiktokenが未対応
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

    def record_ai_completion_stats(self, ctk_prompt, ctk_reply, ctk_cached) -> None:
        now = int(time.time())
        try:
            with sqlite3.connect(_AICHAN_SYSMEMORY_FILE) as conn:
                cur = conn.cursor()
                cur.execute(f"""
                    INSERT INTO {_AICHAN_STATS_TABLE} 
                    (user, class, model, ctk_prompt, ctk_reply, ctk_cached, recorded_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (self.boss_userid, "comp", self.model, ctk_prompt, ctk_reply, ctk_cached, now))
        except Exception as e:
            self.logger.error(f"Error recording stats in record_ai_completion_stats: {e}")

    def record_ai_input_stats(self, systokens, usertokens) -> None:
        now = int(time.time())
        try:
            with sqlite3.connect(_AICHAN_SYSMEMORY_FILE) as conn:
                cur = conn.cursor()
                cur.execute(f"""
                    INSERT INTO {_AICHAN_STATS_TABLE} 
                    (user, class, model, systokens, usertokens, recorded_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (self.boss_userid, "input", self.model, systokens, usertokens, now))
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
                    ORDER BY id
                """, (channel, thread_ts, thread_ts, ts))
                for r_userid, r_content in cur.fetchall():
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

    def prepare_channel_context(self, event: dict) -> list:
        """
        指定チャンネル最近の全履歴を取得し、AI用途に変換する。
        """
        channel = event["channel"]
        conversation_history = []

        try:
            result = self.app.client.conversations_history(channel=channel, limit=30)
            conversation_history = result.get("messages", [])
        except Exception as e:
            self.logger.error("Error in prepare_channel_context: {}".format(e), exc_info=True)

        context = []
        for m in reversed(conversation_history):  # ← 時系列が古い順にしたい場合
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
        ai_messages = [{"role": "system", "content": "\n".join(self.sysmemory)}]

        if in_thread:
            user_messages = self.prepare_thread_context(event)
        else:
            user_messages = self.prepare_channel_context(event)

        # input tokens
        systokens = self.num_tokens_from_messages(ai_messages)
        usertokens = self.num_tokens_from_messages(user_messages)
        self.record_ai_input_stats(systokens, usertokens)
        
        ai_messages += user_messages
        try:
            response = self.ai.chat.completions.create(
                model=self.model,
                messages=ai_messages
            )
            ai_response = response.choices[0].message.content
            ai_response = self.markdown_to_slack(ai_response)
            
            # record completion tokens
            ctk_prompt = response.usage.prompt_tokens
            ctk_reply = response.usage.completion_tokens
            ctk_cached = response.usage.prompt_tokens_details.cached_tokens
            self.record_ai_completion_stats(ctk_prompt, ctk_reply, ctk_cached)
            
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

    def cmd_replace_sysprompt(self, body, say, respond):
        user_id = body.get("user_id")
        channel_id = body.get("channel_id")
        text = body.get("text", "").strip()  # 新しいプロンプト内容

        if not text:
            respond("新しいシステムプロンプトを入力してください。例: `/ai_replace_sysprompt あなたはPythonのプロです。`")
            return

        try:
            with sqlite3.connect(_AICHAN_SYSMEMORY_FILE) as conn:
                cur = conn.cursor()
                cur.execute(f"""
                    UPDATE {_AICHAN_SYSMEMORY_TABLE} 
                    SET content = ?
                    WHERE user = ?
                    """, (text, self.boss_userid))

            self.load_sysmemory()

            respond(f"【システムプロンプトを更新しました】\n{text}")
        except Exception as e:
            self.logger.error(f"sysprompt更新エラー：{e}")
            respond("システムプロンプトの更新に失敗しました。")

    def cmd_add_sysprompt(self, body, say, respond):
        text = body.get("text", "").strip()  # 追加するプロンプト内容

        if not text:
            respond("追加するシステムプロンプトを入力してください。例: `/ai_add_sysprompt プロジェクトはテスト段階に移行した。`")
            return

        additional_text ="\n\n(追記)\n\n" + text

        try:
            with sqlite3.connect(_AICHAN_SYSMEMORY_FILE) as conn:
                cur = conn.cursor()
                # 例: ユーザーごとに追記したい場合はUNIQUEキーなどに注意してください
                cur.execute(f"""
                    INSERT INTO {_AICHAN_SYSMEMORY_TABLE} (user, content)
                    VALUES (?, ?)
                    """, (self.boss_userid, additional_text))

            self.load_sysmemory()

            respond(f"【新しいシステムプロンプトを追加しました】\n{text}")
        except sqlite3.IntegrityError:
            respond("同じユーザーのシステムプロンプトは既に存在します。上書きの場合は `/ai_replace_sysprompt` をご利用ください。")
        except Exception as e:
            self.logger.error(f"sysprompt追加エラー：{e}")
            respond("システムプロンプトの追加に失敗しました。") 

    def cmd_choose_model(self, body, say, respond):
        # GPT-4.1|GPT-4.1-mini|4.1-nano|DALL-E-3
        model = body.get("text", "").strip()

        if model == "gpt-4.1":
            self.model = "gpt-4.1"
        elif model == "gpt-4.1-mini":
            self.model = "gpt-4.1-mini"
        elif model == "gpt-4.1-nano":
            self.model = "gpt-4.1-nano"
        elif model == "dall-e-3":
            self.model = "dall-e-3"
        else:
            respond("no such model")
            return

        previous = self.model
        self.model = model
        respond(f"モデルを{previous}から{model}に変更しました")


    def cmd_token_stats(self, body, say, respond):
        """
        トークン数に関する統計を表示する
        """
        def calc_fee(prompt, reply, cached):
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

        model = self.model
        if model == "gpt-4.1":
            input_price, cached_price, output_price = 2, 0.5, 8
        elif model == "gpt-4.1-mini":
            input_price, cached_price, output_price = 0.4, 0.1, 1.6
        elif model == "gpt-4.1-nano":
            input_price, cached_price, output_price = 0.1, 0.025, 0.4
        elif model == "dall-e-3": #統計は未実装だけど。
            price_1024, price_1792 = 0.04, 0.08
        else:
            respond(f"modelがおかしいゾ？ {model=}")
            return

        # 価格は1M Tokenあたりの単価なので、1 Tokenあたりに換算
        input_price = input_price / 1000000
        cached_price = cached_price / 1000000
        output_price = output_price / 1000000

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
                    WHERE user = ? AND class = ?
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
                    fee = calc_fee(prompt, reply, cached)
                    total_fee += fee
                    response.append(f"{recorded_at:<20} {model:<15} {prompt:>8} {reply:>8} {cached:>8} {format_usd(fee)}")
                response.append("```")  # コードブロックを閉じる

                # 2. それ以降のレコードは集計だけ
                for model, prompt, reply, cached, recorded_at in rows[10:]:
                    fee = calc_fee(prompt, reply, cached)
                    total_fee += fee

                # 合計金額
                response.append(f"\nAccumulated Fee : USD {total_fee:.2f} (JPY {total_fee * usd_to_jpy:,.0f}円@{usd_to_jpy}) since {rows[-1][4]}")
                respond("\n".join(response))
        except Exception as e:
            print(f"DB error: {e}")
            respond("統計情報の取得に失敗しました")


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

@app.command("/ai_add_sysprompt")
def handle_add_sysprompt(ack, body, say, respond):
    """
    システムプロンプトを新規追加するスラッシュコマンド
    使い方： /ai_add_sysprompt 新規プロンプト内容
    """
    ack()
    aichan.cmd_add_sysprompt(body, say, respond)

@app.command("/ai-model")
def handle_model(ack, body, say, respond):
    ack()
    aichan.cmd_choose_model(body, say, respond)

@app.command("/ai-tokens")
def handle_tokens(ack, body, say, respond):
    ack()
    aichan.cmd_token_stats(body, say, respond)

if __name__ == "__main__":
    handler = SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"])
    handler.start()        
