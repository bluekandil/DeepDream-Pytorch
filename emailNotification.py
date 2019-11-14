import smtplib, ssl

def notify():
    port = 465  # For SSL
    smtp_server = "smtp.gmail.com"
    sender_email = "sharelov365@gmail.com"  # Enter your address
    receiver_email = "sharelov365@gmail.com"  # Enter receiver address
    password ="@Chigu@2019"
    message = """\
    Subject: Hi there

    Tweet unable to post."""

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message)

