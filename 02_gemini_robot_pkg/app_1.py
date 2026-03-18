from __future__ import annotations
import html
import secrets
import sqlite3
import urllib.parse
from http import cookies
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# 경로 설정
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "users.db"
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

SESSIONS: dict[str, dict[str, str]] = {}

# 시스템 기능 목록
FEATURES = {
    "overview": {"title": "시스템 개요", "description": "RealSense/C270 입력 및 로봇 제어 흐름 확인.", "items": ["카메라 입력 상태", "현재 공정 단계", "오류 상태"]},
    "eye": {"title": "눈 (eye.py)", "description": "YOLO 기반 객체 감지 및 좌표 생성.", "items": ["부품 감지", "드라이버 감지", "좌표 생성"]},
    "brain": {"title": "뇌 (brain.py)", "description": "Gemini AI 기반 작업 순서 결정.", "items": ["회로도 해석", "음성 명령 결합", "좌표 정규화"]},
    "voice": {"title": "귀/입 (voice)", "description": "STT/TTS를 통한 사용자 상호작용.", "items": ["음성 인식", "음성 출력", "명령 체크"]},
    "robot": {"title": "신경/근육 (robot)", "description": "물리적 모션 수행 및 로깅.", "items": ["역정규화", "모션 실행", "로그 저장"]},
    "progress": {"title": "공정 관리", "description": "전체 작업 진행률 계산.", "items": ["전선 감지", "진행률 퍼블리시"]},
}

def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE NOT NULL, password TEXT NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
    conn.commit()
    conn.close()

def get_template(name: str) -> str:
    file_path = TEMPLATES_DIR / name
    return file_path.read_text(encoding="utf-8") if file_path.exists() else f"Template {name} not found."

def render(template_name: str, context: dict) -> str:
    content = get_template(template_name)
    for key, value in context.items():
        content = content.replace(f"{{{{{ key }}}}}", str(value))
    return content

class Handler(BaseHTTPRequestHandler):
    def current_user(self) -> dict[str, str] | None:
        raw = self.headers.get("Cookie", "")
        jar = cookies.SimpleCookie()
        jar.load(raw)
        sid = jar["sid"].value if "sid" in jar else ""
        return SESSIONS.get(sid)

    def send_html(self, html_text: str, code: int = 200, set_cookie: str | None = None) -> None:
        payload = html_text.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        if set_cookie: self.send_header("Set-Cookie", set_cookie)
        self.end_headers()
        self.wfile.write(payload)

    def redirect(self, location: str, set_cookie: str | None = None) -> None:
        self.send_response(302)
        self.send_header("Location", location)
        if set_cookie: self.send_header("Set-Cookie", set_cookie)
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)
        message = query.get("msg", [""])[0]
        user = self.current_user()

        # 정적 파일 처리
        if path.startswith("/static/"):
            file_path = STATIC_DIR / path[len("/static/"):]
            if file_path.exists() and file_path.is_file():
                data = file_path.read_bytes()
                self.send_response(200)
                if path.endswith(".css"): self.send_header("Content-Type", "text/css")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            self.send_error(404)
            return

        # 페이지 렌더링 로직
        top_right = ""
        if user:
            top_right = f"<div class='topbar-right'><span class='username'>{html.escape(user['username'])} 님</span><form method='post' action='/logout'><button class='outline' type='submit'>로그아웃</button></form></div>"
        
        flash = f"<ul class='flash-list'><li>{html.escape(message)}</li></ul>" if message else ""

        if path == "/":
            self.redirect("/dashboard" if user else "/login")
        elif path == "/login":
            body = get_template("login.html")
            self.send_html(render("layout.html", {"title": "로그인", "top_right": "", "flash": flash, "body": body}))
        elif path == "/signup":
            body = get_template("signup.html")
            self.send_html(render("layout.html", {"title": "회원가입", "top_right": "", "flash": flash, "body": body}))
        elif path == "/dashboard":
            if not user: self.redirect("/login?msg=로그인이 필요합니다")
            else:
                cards = "".join(f"<article class='card feature-card'><h3>{v['title']}</h3><p>{v['description']}</p><a class='button-link' href='/feature/{k}'>열기</a></article>" for k, v in FEATURES.items())
                body = render("dashboard.html", {"cards": cards})
                self.send_html(render("layout.html", {"title": "대시보드", "top_right": top_right, "flash": flash, "body": body}))
        elif path.startswith("/feature/"):
            name = path.split("/")[-1]
            f = FEATURES.get(name)
            if not f: self.redirect("/dashboard")
            else:
                li = "".join(f"<li>{item}</li>" for item in f["items"])
                body = render("feature.html", {"title": f["title"], "description": f["description"], "items": li})
                self.send_html(render("layout.html", {"title": f["title"], "top_right": top_right, "flash": flash, "body": body}))
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        # 기존 POST 로직(signup, login, logout) 유지 (생략)
        pass

if __name__ == "__main__":
    init_db()
    print("Server running on http://localhost:5000")
    ThreadingHTTPServer(("0.0.0.0", 5000), Handler).serve_forever()
