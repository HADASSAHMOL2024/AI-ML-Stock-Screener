import os
import webbrowser
from flask import Flask, request
from dotenv import load_dotenv
from fyers_apiv3 import fyersModel

load_dotenv()

CLIENT_ID = os.getenv("FYERS_CLIENT_ID")
SECRET_ID = os.getenv("FYERS_SECRET_ID")
REDIRECT_URI = os.getenv("FYERS_REDIRECT_URI")

app = Flask(__name__)

# Create FYERS authentication session
session = fyersModel.SessionModel(
    client_id=CLIENT_ID,
    secret_key=SECRET_ID,
    redirect_uri=REDIRECT_URI,
    response_type="code",
    grant_type="authorization_code",
    state="stock_screener"
)

auth_url = session.generate_authcode()


@app.route("/")
def home():
    return """
    <h2>AI/ML Stock Market Screening</h2>
    <p>Authentication server is running.</p>
    <p>Complete FYERS login in the browser.</p>
    """


@app.route("/callback")
def callback():
    auth_code = request.args.get("auth_code")

    if not auth_code:
        return """
        <h2>Authentication failed</h2>
        <p>No authorization code was received.</p>
        """

    try:
        session.set_token(auth_code)

        response = session.generate_token()

        if "access_token" not in response:
            return f"""
            <h2>Token generation failed</h2>
            <pre>{response}</pre>
            """

        access_token = response["access_token"]

        with open("access_token.txt", "w") as file:
            file.write(access_token)

        return """
        <h2>FYERS authentication successful!</h2>
        <p>You can close this browser tab.</p>
        <p>Your access token has been saved locally.</p>
        """

    except Exception as e:
        return f"""
        <h2>Authentication error</h2>
        <pre>{e}</pre>
        """


if __name__ == "__main__":
    print("=" * 60)
    print("FYERS Authentication")
    print("=" * 60)

    print("\nOpening FYERS login page...")

    webbrowser.open(auth_url)

    print("\nComplete the FYERS login in your browser.")
    print("Waiting for authentication callback...\n")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )