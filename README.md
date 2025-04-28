# AIちゃん (Aichan - Slack Bot for OpenAI)

Slack 上で OpenAI API を活用する、会話型 AI ボットです。

会話履歴やシステムプロンプトの記録、統計管理、スレッドごとの文脈保持などを特徴としています。

Slack Bolt + OpenAI + SQLite3 によるシンプルかつ柔軟な構成です。

## 構成ファイル
- app.py : メインアプリケーション。Slackイベントやコマンド処理のロジックを含みます。
- requirements.txt : 必要な Python パッケージ一覧。

## 主な機能
- GPT-4.1（nano/mini含む）によるスレッド文脈を加味した対話
- Slackチャンネルごとのモデル切替対応（gpt-4.1, gpt-4.1-mini, gpt-4.1-nano）
- Slackチャンネルごとのシステムプロンプト（スラッシュコマンドで設定可能）
- SQLite による会話履歴と統計の保存
- メンションやスレッド内での応答、時々発話機能
- トークン使用量と想定コストを表示（USD/JPY換算）

## セットアップ方法
1.	リポジトリをクローン

```bash
git clone https://github.com/shunji-taki/aichan.git
cd aichan
```

2.	Python 仮想環境を作成し、依存関係をインストール

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3.	.env ファイルを作成し、以下を記述

```env
SLACK_BOT_TOKEN=（Slack Bot Token）
SLACK_APP_TOKEN=（Socket Mode Token）
OPENAI_API_KEY=（OpenAI API Key）
BOSS_SLACK_USERID=（オーナーのSlack User ID）
```

4.	アプリ起動

```bash
python app.py
```

## スラッシュコマンド一覧
- /ai_replace_sysprompt : システムプロンプトを新しい内容に置き換える
- /ai_set_persona       : チャンネルでのペルソナを設定する
- /ai-model             : チャンネルでのOpenAIモデルを変更する
- /ai-tokens            : トークン使用状況とコスト統計を表示する
- /ai-tweet             : ときどき発話する機能を設定する

## データファイル（SQLite）
- conversation.db : 会話ログの記録
- sysmemory.db    : システムプロンプトやトークン統計の保存

## 依存ライブラリ（抜粋）
- slack-bolt
- openai
- python-dotenv
- tiktoken

## ライセンス

本プロジェクトのライセンスは未定です。再利用・商用利用については著作権者の許可を得てください。