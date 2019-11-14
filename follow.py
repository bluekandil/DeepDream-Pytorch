import tweepy 
import time


CONSUMER_KEY='XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
CONSUMER_SECRET='XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
ACCESS_TOKEN='XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
ACCESS_TOKEN_SECRET='XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'


auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
api = tweepy.API(auth)

# for follower in tweepy.Cursor(api.search, q="#joker").items():    
#     api.create_friendship(screen_name = follower.author.screen_name)
#     print(follower.author.screen_name)

def get_followers():
    users = []
    page_count = 2
    for i, user in enumerate(tweepy.Cursor(api.followers, id="deepdreamartbot", count=500).pages()):
        # print "Getting page {} for followers"+format(i)
      
        users += user

        # -----------------------------------------------------------------------
        
        friends = api.friends_ids(api.me().id)
        for u in users:
            follower=u
            for follower in tweepy.Cursor(api.followers).items():
                if follower.id != api.me().id:
                    if follower.id in friends:
                        print("You already follow", follower.screen_name)
                    else:
                        try:
                            follower.follow()
                            print("Started following", follower.screen_name)

                        except:
                            pass
                        
            






        # -----------------------------------------------------------------------
        # print(users.screen_name)
        # users[0]
        # print(users[0].screen_name)
        user_b='fetchdeepdream'
        for u in users:
            print(u.screen_name)
            try:
                if(api.exists_friendship(user_b,u.screen_name)):
                    print('already...')
                # api.create_friendship(screen_name =u.screen_name)
                # print("followed")
            except:
                api.create_friendship(screen_name =u.screen_name)
                #  print("can not ...")
            
    # return users
get_followers()


# for follower in tweepy.Cursor(api.followers).items():
#     if follower.id != api.me().id:
#         if follower.id in friends:
#             print("You already follow", follower.screen_name)
#         else:
#             follower.follow()
#             print("Started following", follower.screen_name)
