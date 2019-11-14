import tweepy 
import os
import emailNotification
# Twitter Auth

consumer_key='XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
consumer_secret='XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
access_token='XXXXXXXXXXXXX-XXXXXXXXXXXXXXXXXXX'
access_token_secret='XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

try:
    api = tweepy.API(auth)
    user = api.me()
    print("Authenticated ..."+user.name)
except tweepy.TweepError:
    print('Error!..Authentication failed')
    emailNotification.notify()
    


# Post Image
def postTweetImage(msg,description,img_path):
    
    fileName=msg
    description=description
    # os.path.splitext(msg)[0]
    
    tweet ="#DeepDream"+" "+"#DeepDreamGallery"+"\n"+"#Pytorch"+" "+"#Digitalart"+" "+"\n"+"#"+fileName+"  "+description
    # + "#firsttweet"  
 
    image_path =img_path
    # "img/cat1_trippy.jpg" # toDo 
    # to attach the media file 
    try:
        status = api.update_with_media(image_path,tweet)
        print("posted.....") 
    except tweepy.TweepError:
        print('Error!..')
 
    # api.update_status(status = tweet)
