from Crypto.Cipher import AES
from Crypto.Protocol.KDF import PBKDF2
import os
import tempfile
from dotenv import load_dotenv

load_dotenv()  # Load all environment variables

def unpad(data):
    return data[:-data[-1]]

def decrypt_and_run():
    # Get password from Hugging Face Secrets environment variable
    password = os.getenv("PASSWORD")
    if not password:
        raise ValueError("PASSWORD secret not found in environment variables")

    password = password.encode()

    with open("code.enc", "rb") as f:
        encrypted = f.read()

    salt = encrypted[:16]
    iv = encrypted[16:32]
    ciphertext = encrypted[32:]

    key = PBKDF2(password, salt, dkLen=32, count=1000000)
    cipher = AES.new(key, AES.MODE_CBC, iv)

    plaintext = unpad(cipher.decrypt(ciphertext))

    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode='wb') as tmp:
        tmp.write(plaintext)
        tmp.flush()
        print(f"[INFO] Running decrypted code from {tmp.name}")
        os.system(f"python {tmp.name}")

if __name__ == "__main__":
    decrypt_and_run()

# This script decrypts the encrypted code and runs it.
# Ensure you have the PASSWORD secret set in your Hugging Face Secrets