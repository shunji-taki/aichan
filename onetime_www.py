import os
import logging
from io import BytesIO

import requests
from fastapi import FastAPI, Response
import uuid, os
from PIL import Image

from dotenv import load_dotenv

load_dotenv()

class BotConfigError(Exception):
    """OneTimeWWWが送出するコンフィグ関連の例外"""

class OneTimeWWW:
    """OpenAI FileAPIにファイルを渡す実装"""
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            loghandler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            loghandler.setFormatter(formatter)
            self.logger.addHandler(loghandler)

        self.tmp_dir = "/tmp/image_files"
        os.makedirs(self.tmp_dir, exist_ok=True)
        self.img_db = {}  # img_id: filepath

        self.fileurl = os.environ.get("BOT_FILESERVER_BASEURL")
        if not self.fileurl:
            raise BotConfigError("環境変数BOT_FILESERVER_BASEURLを取得できません")

    def download_slack_file(self, url_private, token):
        """Slack APIからファイルをダウンロード"""
        headers = {'Authorization': f'Bearer {token}'}
        response = requests.get(url_private, headers=headers, timeout=10)
        response.raise_for_status()
        return response.content

    def process_image(self, src_bytes, max_size=2000):
        """画像ファイルの圧縮"""
        im = Image.open(BytesIO(src_bytes))
        im = im.convert("RGB")
        # サイズ超過なら縮小
        if max(im.size) > max_size:
            ratio = max_size / max(im.size)
            newsize = (int(im.size[0]*ratio), int(im.size[1]*ratio))
            im = im.resize(newsize)
        buf = BytesIO()
        im.save(buf, format="JPEG", quality=85)  # 必要に応じてPNGも可
        return buf.getvalue()

    def serve_once(self, img_id: str):
        """getされたらファイルを消す"""
        fname = self.img_db.pop(img_id, None)
        if fname and os.path.exists(fname):
            with open(fname, "rb") as f:
                content = f.read()
            os.remove(fname)  # 物理削除
            return Response(content, media_type="image/png")
        else:
            return Response("Not Found", status_code=404)

    def generate_onetime_url(self, image_bytes: bytes) -> str:
        """ワンタイムURLを生成"""
        img_id = str(uuid.uuid4())
        fname = os.path.join(self.tmp_dir, f"{img_id}.png")
        with open(fname, "wb") as f:
            f.write(image_bytes)
        self.img_db[img_id] = fname
        return f"http://{self.fileurl}/img-once/{img_id}"

webapp = FastAPI()
onetime_www = OneTimeWWW()

@webapp.get("/img-once/{img_id}")
def get_called(img_id: str):
    """get処理"""
    return onetime_www.serve_once(img_id)
