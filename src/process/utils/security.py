"""
Create on 2025/8/12

@auther:Thinkpad
"""
import os
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import base64
import logging

logger = logging.getLogger("security")


class SecureProcessor:
    def __init__(self, encryption_key=None):
        """初始化安全处理器"""
        self.salt = os.urandom(16)

        if encryption_key:
            if isinstance(encryption_key, str):
                encryption_key = encryption_key.encode()
        else:
            encryption_key = Fernet.generate_key()

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self.salt,
            iterations=100000,
            backend=default_backend()
        )
        self.encryption_key = base64.urlsafe_b64encode(kdf.derive(encryption_key))
        self.cipher = Fernet(self.encryption_key)

    def encrypt_file(self, input_path, output_path=None):
        """加密文件"""
        if not output_path:
            output_path = input_path + ".enc"

        try:
            with open(input_path, 'rb') as f:
                data = f.read()

            encrypted = self.cipher.encrypt(data)

            with open(output_path, 'wb') as f:
                f.write(encrypted)

            return output_path
        except Exception as e:
            logger.error(f"文件加密失败: {str(e)}")
            raise

    def decrypt_file(self, input_path, output_path=None):
        """解密文件"""
        if not output_path:
            output_path = input_path.replace(".enc", "")

        try:
            with open(input_path, 'rb') as f:
                encrypted = f.read()

            decrypted = self.cipher.decrypt(encrypted)

            with open(output_path, 'wb') as f:
                f.write(decrypted)

            return output_path
        except Exception as e:
            logger.error(f"文件解密失败: {str(e)}")
            raise

    def clean_secure_tempfiles(self, temp_dir=None):
        """清理临时文件"""
        import tempfile
        import glob

        target_dir = temp_dir or tempfile.gettempdir()
        for file in glob.glob(os.path.join(target_dir, "*.enc")):
            try:
                os.remove(file)
            except:
                pass

