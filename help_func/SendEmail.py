import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from help_func.logging import LoggingHelper
import os


def attachEmail(path, msg):
    attachment = open(path, 'rb')
    part = MIMEBase('application', 'octet-stream')
    part.set_payload((attachment).read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', "attachment; filename= " + os.path.basename(path))
    msg.attach(part)


def SendEmail(subject, body, ToMailAdress, attachedFiles=None):
    try:

        s = smtplib.SMTP('smtp.gmail.com', 587)

        s.starttls()

        s.login('tuzm24@gmail.com', 'jhexndngsgqhamdg')

        msg = MIMEMultipart()
        msg['Subject'] = subject
        msg['From'] = 'tuzm24@gmail.com'
        msg['To'] = ToMailAdress
        msg.attach(MIMEText(body, 'plain'))
        if attachedFiles is not None:
            if isinstance(attachedFiles, list):
                for file in attachedFiles:
                    attachEmail(file, msg)
            else:
                attachEmail(attachedFiles, msg)

        s.sendmail('tuzm24@gmail.com', ToMailAdress, msg.as_string())

        # s.sendmail("tuzm24@gmail.com", "leap1568@gmail.com", msg.as_string())
        # s.sendmail("tuzm24@gmail.com", "nukim@sju.ac.kr", msg.as_string())

        s.quit()
    except Exception as e:
        logger = LoggingHelper.get_instance().logger
        logger.error('Fail to send Email.. : %s' %e)



# SendEmail('test', 'test', 'ywkim@sju.ac.kr', ['C:/Users/YangwooKim/Desktop/e4cd68968b7e4c2eb73.jpg'])