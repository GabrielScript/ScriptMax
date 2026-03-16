import smtplib
import os
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from dotenv import load_dotenv

load_dotenv()

class EmailSender:
    def __init__(self):
        self.smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.email_user = os.getenv("EMAIL_USER")
        self.email_password = os.getenv("EMAIL_PASSWORD") # This must be an App Password
        self.default_recipient = os.getenv("DEFAULT_RECIPIENT", "gabrielestrela8@gmail.com")

    def send_report(self, subject, html_path, pdf_path, recipient=None):
        """Sends an email with the HTML and PDF reports as attachments."""
        if not self.email_user or not self.email_password:
            print("❌ Erro: Configurações de e-mail ausentes no .env")
            return False

        recipient = recipient or self.default_recipient
        
        msg = MIMEMultipart()
        msg['From'] = self.email_user
        msg['To'] = recipient
        msg['Subject'] = f"🎓 ScriptMax: Relatório de Aula - {subject}"

        body = f"Olá!\n\nSeu relatório para a aula '{subject}' foi gerado com sucesso.\n\nEm anexo você encontrará a versão em PDF (com fórmulas) e a versão em HTML.\n\nAtenciosamente,\nEquipe ScriptMax"
        msg.attach(MIMEText(body, 'plain'))

        # Attach PDF
        if pdf_path and os.path.exists(pdf_path):
            with open(pdf_path, "rb") as attachment:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment.read())
                encoders.encode_base64(part)
                part.add_header('Content-Disposition', f"attachment; filename= {os.path.basename(pdf_path)}")
                msg.attach(part)

        # Attach HTML
        if html_path and os.path.exists(html_path):
            with open(html_path, "rb") as attachment:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment.read())
                encoders.encode_base64(part)
                part.add_header('Content-Disposition', f"attachment; filename= {os.path.basename(html_path)}")
                msg.attach(part)

        try:
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.email_user, self.email_password)
            text = msg.as_string()
            server.sendmail(self.email_user, recipient, text)
            server.quit()
            print(f"✅ E-mail enviado com sucesso para {recipient}")
            return True
        except Exception as e:
            print(f"❌ Erro ao enviar e-mail: {e}")
            return False
