# %% [markdown]
# # Pikmin
# 
# __Project Overview__
# 
# Goal: Determine 'best' pikmin based on three metrics
# - Cultural impact of pikmin
#     - Via # of Reddit posts on r/Pikmin and r/PikminBloomApp
# - Public outlook on pikmin
#     - Via average # of upvotes on Reddit posts
#     - Via NLP processing of comments on these posts determining 'positive' feedback
# - Viability of pikmin in game
#     - Via stats from Pikmin 4 via Pikipedia
# 
# Secondary goal: Determine which pikmin each of my friends is most like
# - From MBTI (self reported from friends)
#     - Scrape 16personalities for top traits of each personality
#     - Generate similarity score between traits of personality and adjectives from Reddit comments
# 

# %% [markdown]
# __Reddit API use with PRAW__

# %%
# libraries
import requests
import praw
import pandas as pd
import re
import matplotlib.pyplot as plt
from collections import Counter
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import bs4
import statistics
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# nltk.download('vader_lexicon')


# %%
reddit = praw.Reddit(
    client_id='client_id',
    client_secret='client_secret',
    user_agent='pikmin'
) # please get your own key

# %%
pikmin = reddit.subreddit('pikmin')
pikmin_bloom = reddit.subreddit('pikminbloomapp')
r_all = reddit.subreddit('all')

print(pikmin.title)

# %%
pikmin_colors = ["red", "yellow", "blue", "pink", "rock", "white", "ice", "purple"]

posts_df = pd.DataFrame()

titles = []
scores = []
ids = []
colors = []

for submission in pikmin.top(limit=None):
    title = str(submission.title)
    title = ''.join(re.findall(r'[a-zA-Z ]', title))
    title_split = title.split(' ')

    for color in pikmin_colors:
        for word in title_split:
            if color == word:
                titles.append(submission.title)
                scores.append(submission.score)
                ids.append(submission.id)
                colors.append(color)
                break

for submission in pikmin_bloom.top(limit=None):
    title = str(submission.title)
    title = ''.join(re.findall(r'[a-zA-Z ]', title))
    title_split = title.split(' ')

    for color in pikmin_colors:
        for word in title_split:
            if color == word:
                titles.append(submission.title)
                scores.append(submission.score)
                ids.append(submission.id)
                colors.append(color)
                break

posts_df['title'] = titles
posts_df['id'] = ids
posts_df['score'] = scores
posts_df['color'] = colors

print(posts_df.shape)
posts_df.head()

# %%
counter_dict = Counter(list(posts_df['color']))
counter_dict = dict(sorted(counter_dict.items(), key=lambda item: item[1], reverse=True))
colors = list(counter_dict.keys())
frequencies = list(counter_dict.values())
plt.grid(True, axis='y', linestyle = '--', alpha = 0.4)
plt.bar(colors, frequencies, color = ['cornflowerblue', 'lightgray', 'plum', 'gold', 'gray', 'lightcoral', 'lightblue', 'pink'], edgecolor='darkslategray')
plt.xlabel('pikmin color')
plt.ylabel('number of posts')
plt.title('posts per pikmin color')
plt.show()


# %%
mean_score_by_color = posts_df.groupby('color')['score'].mean()

df_list = []
for color in pikmin_colors:
    df = posts_df['score'][posts_df['color']==color]
    df_list.append(df)

colors = ['lightcoral', 'gold', 'cornflowerblue', 'pink', 'gray', 'lightgray', 'lightblue', 'plum']

box = plt.boxplot(df_list, tick_labels = pikmin_colors, patch_artist = True)
plt.grid(True, axis='y', linestyle = '--', alpha = 0.4)

for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_edgecolor('darkslategray')
    patch.set_linewidth(1.5)

plt.xlabel('pikmin color')
plt.ylabel('upvotes')
plt.title('post performance by pikmin color')

plt.show()

# %%
def analyze_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(text)
    compound_score = sentiment_scores['compound']
    
    if compound_score >= 0.05:
        return "Positive"
    elif compound_score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

df_list = []
for color in pikmin_colors:
    df = posts_df[posts_df['color']==color]
    df_list.append(df)

colors = []
sentiments = []
for color_df in df_list:
    color = list(color_df['color'])[0]

    for url in list(color_df['id']):
        submission = reddit.submission(url)

        for comment in submission.comments:
            body = str(comment.body)
            sentiment = analyze_sentiment(body)

            colors.append(color)
            sentiments.append(sentiment)

sentiment_df = pd.DataFrame({
    'color': colors,
    'sentiment': sentiments
})

print(sentiment_df.shape)
sentiment_df.head()

# %%
sentiment_counts = sentiment_df.groupby(['color', 'sentiment']).size().unstack(fill_value=0)
sentiment_percentages = sentiment_counts.div(sentiment_counts.sum(axis=1), axis=0) * 100

ax = sentiment_percentages.plot(kind='bar', stacked=True, color=['firebrick', 'darkgray', 'green'])
plt.xlabel('pikmin color')
plt.ylabel('percentage')
plt.title('sentiment distribution per pikmin color')

plt.show()

# %%
sentiment_percentages.head()

# %% [markdown]
# __Pikipedia Scraping__

# %%
pikmin_urls = ['Red_Pikmin', 'Yellow_Pikmin', 'Blue_Pikmin', 'Purple_Pikmin', 'White_Pikmin', 'Rock_Pikmin', 'Winged_Pikmin', 'Ice_Pikmin']

pikmin_stats = pd.DataFrame({
    'color': [],
    'atk_strength': [],
    'carry_strength': [],
    'run_speed': [],
    'dig_speed': []
})
for url in pikmin_urls:
    full_url = 'https://www.pikminwiki.com/' + url
    response = requests.get(full_url)
    clean_response = bs4.BeautifulSoup(response.text, 'html.parser')

    tables = clean_response.find_all('table')
    technical_tables = [table for table in tables if 'technicaltable' in str(table) and '/Pikmin_4' in str(table)]
    pikmin4_table = bs4.BeautifulSoup(str(technical_tables[0]), 'html.parser')

    data = [url]
    table_rows = pikmin4_table.find_all('tr')
    for row in table_rows:
        for cell in row.find_all('td'):
            text = cell.text
            num_only = re.sub(r'[^0-9,.]', '', text)
            values = num_only.split(',')
            i = 0
            for value in values:
                try: float(value)
                except: fl_value = 0
                else: fl_value = float(value)
                values[i] = fl_value
                i += 1
            mean_stat = statistics.mean(values)

            data.append(mean_stat)

    pikmin_stats.loc[len(pikmin_stats)] = data

print(pikmin_stats)
    

# %%
# since rock pikmin atk strength is listed as 'depends' we will calculate it based on a ratio between its pikmin bloom attack strength and purple/blue pikmin's attack strength (average of both)
# x/5 = 15/6 and x/5 = 10/3

rock_pikmin_strength = round((((5*15)/6) + ((5*10)/3))/2,1)

pikmin_stats.at[5, 'atk_strength'] = rock_pikmin_strength

print(pikmin_stats)

# %%
columns = ['atk_strength', 'carry_strength', 'run_speed', 'dig_speed']

pikmin_stats_standard = pd.DataFrame({
    'color': list(pikmin_stats['color'])
})
for col in columns:
    cur_list = list(pikmin_stats[col])
    max_value = max(cur_list)
    new_list = []
    for item in cur_list:
        new_item = (item / max_value) * 10
        new_list.append(new_item)
    pikmin_stats_standard[col] = new_list

print(pikmin_stats_standard)

# %%
print(pikmin_stats_standard.iloc[0].tolist())

# %%
data = [
    pikmin_stats_standard.iloc[0].tolist(),
    pikmin_stats_standard.iloc[1].tolist(),
    pikmin_stats_standard.iloc[2].tolist(),
    pikmin_stats_standard.iloc[3].tolist(),
    pikmin_stats_standard.iloc[4].tolist(),
    pikmin_stats_standard.iloc[5].tolist(),
    pikmin_stats_standard.iloc[6].tolist(),
    pikmin_stats_standard.iloc[7].tolist()
]

num_vars = len(data[0]) - 1 

angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

angles += angles[:1]

fig, axs = plt.subplots(2, 4, figsize=(15, 10), subplot_kw=dict(polar=True))

colors = ['lightcoral', 'gold', 'cornflowerblue', 'plum', 'lightgray', 'gray', 'pink', 'lightblue']

for i, (name, stat1, stat2, stat3, stat4) in enumerate(data):
    stats = [stat1, stat2, stat3, stat4]
    
    stats += stats[:1]
    
    ax = axs[i // 4, i % 4]  
    
    ax.plot(angles, stats, linewidth=2, linestyle='solid')
    ax.fill(angles, stats, alpha=0.4, color=colors[i], edgecolor='darkslategray')
    
    ax.set_yticklabels([]) 
    ax.set_xticks(angles[:-1]) 
    ax.set_xticklabels(['atk', 'carry', 'run', 'dig'], fontsize=10)
    ax.set_title(name, size=12)
    ax.set_ylim(0, 10)

plt.tight_layout()
plt.show()

# %% [markdown]
# __Calculating a score for each Pikmin (Standardization)__
# 
# In each metric, pikmin will be given a score out of 10 that will be 10 = the highest score in that metric, and every other score scaled to that
# 
# Then each category will sum each metric and take the results / 10
# 
# Then each category will be summed and the final score for each pikmin will be out of 10 (/3)

# %%
# Cultural impact: # of Reddit posts
pik_colors = list(counter_dict.keys())
num_posts = list(counter_dict.values())
max_val = max(num_posts)

num_posts_std = []
for num in num_posts:
    new_num = (num / max_val) * 10
    num_posts_std.append(new_num)

pikmin_scores_df = pd.DataFrame({
    'color': pik_colors,
    'cultural_impact': num_posts_std
})

print(pikmin_scores_df)


# %%
# Public outlook: Reddit karma and comment sentiment

# karma
scores = list(mean_score_by_color)
colos = ['blue', 'ice', 'pink', 'purple', 'red', 'rock', 'white', 'yellow']
max_val = max(scores)
new_vals = []
for val in scores:
    new_val = (val/max_val) * 10
    new_vals.append(new_val)
karma_df = pd.DataFrame({
    'color': colos,
    'karma': new_vals
})


# sentiment
colos2 = ['blue', 'ice', 'pink', 'purple', 'red', 'rock', 'white', 'yellow']
pos = list(sentiment_percentages['Positive'])
max_val = max(pos)
new_vals = []
for val in pos:
    new_val = (val/max_val) * 10
    new_vals.append(new_val)
pos_sent_df = pd.DataFrame({
    'color': colos,
    'positive': new_vals
})

full_outlook_df = pd.merge(karma_df, pos_sent_df, how='left', on='color')

colos2 = ['blue', 'ice', 'pink', 'purple', 'red', 'rock', 'white', 'yellow']
karma = list(full_outlook_df['karma'])
pos = list(full_outlook_df['positive'])

vals = []
i = 0
for k in karma:
    p = pos[i]
    val = (k + p) / 2
    vals.append(val)
    i += 1

outlook_df = pd.DataFrame({
    'color': colos2,
    'public_outlook': vals
})

print(outlook_df)



# %%
# Viability: Pikipedia stats

pikmin_stats_standard

colos3 = ['red', 'yellow', 'blue', 'purple', 'white', 'rock', 'pink', 'ice']

atk = list(pikmin_stats_standard['atk_strength'])
carry = list(pikmin_stats_standard['carry_strength'])
run = list(pikmin_stats_standard['run_speed'])
dig = list(pikmin_stats_standard['dig_speed'])

stat_vals = []
i = 0
for a in atk:
    c = carry[i]
    r = run[i]
    d = dig[i]

    new_val = (a + c + r + d) / 4
    stat_vals.append(new_val)

    i += 1

viability_df = pd.DataFrame({
    'color': colos3,
    'viability': stat_vals
})

viability_df

# %%
pikmin_scores_df

# %%
pikmin_scores_df = pd.merge(pikmin_scores_df, outlook_df, on='color', how='left')
pikmin_scores_df = pd.merge(pikmin_scores_df, viability_df, on='color', how='left')

cult = list(pikmin_scores_df['cultural_impact'])
pub = list(pikmin_scores_df['public_outlook'])
via = list(pikmin_scores_df['viability'])

overall = []
i = 0
for c in cult:
    p = pub[i]
    v = via[i]

    new_val = (c + p + v) / 3
    overall.append(new_val)

    i += 1

pikmin_scores_df['overall'] = overall

pikmin_scores_df

# %% [markdown]
# __Pikmin personalities__
# 
# https://www.yayomg.com/what-your-favorite-pikmin-says-about-you/

# %%
red_type = '''Fiery and fierce, you’re just like a Red Pikmin! Your friends can count on you to step up – and deliver!

Always ready to lead the pack, your independent spirit and logical mindset fuel your ability to take charge. Whether you’re slaying a group project, leading your sports team to victory, or organizing an epic end-of-summer bash, you’re always fired up when an opportunity to lead comes your way.'''

blue_type = '''Easygoing. Laid back. Totally cool. Ring a bell? If Blue Pikmin are your fav, you’re probably the chill one in your squad, keeping things going swimmingly in the group chat with your quick wit and endless jokes that just hit different.

When plans change, you don’t get caught up in the details – you go with the flow. On land, in water, or in the school caf, your flexible nature comes in handy, helping you to adapt to any situation or pick up new skills like it’s no big thing. Your chill vibes are what draws people to your side, but it’s your personality that really makes a splash!

'''

yellow_type = '''Shockingly great describes Yellow Pikmin, and it describes you, too!

Adventurous and curious, you’re always down to try new things. Something tells us you’re the fun one in your squad, bringing high energy to every activity and ever-ready to seek out a new adventure. Whether it be thrill-seeking jaunts, like riding a coaster with a massive drop or simply convincing your bestie to break out of their comfort zone and try a new restaurant in the next town over, your curiosity electrifies your soul and makes life interesting!

'''

ice_type = '''New to the pack, these chilly Pikmin are already proving they’re as dependable as the OGs. So how are you like these icy creatures? You’re always there when you say you will be – sitting front row at your bestie’s acting debut in the school musical, dropping everything to craft a pick-me-up for a friend who’s going through it, or sacrificing weekend fun to help mom re-paint the living room – you name it.

You can always sense what really matters to the people around you and prioritize their needs, and it’s the coolest thing about you!'''

purple_type = '''Strong, yet fabulous. If this sounds like you, then it’s the Purple Pikmin you relate to most, for sure. These grape-colored creatures are as tough as you are confident!

Big test? Team captain? Karaoke night? You’re not nervous at all, in fact, you thrive in any situation where you can give it your all and do your best. Handling pressure comes naturally to you, and you’re always stunning others with your ability to shake off everything from a bad day to a bad mood with ease.'''

rock_type = '''Always ready to stand up for what you believe in, there’s no surprise the Rock Pikmin is your fav. They’re sturdy and unbreakable, just like your plucky, Pikmin-like spirit! You have a tendency to jump in and speak your mind, especially when your core beliefs are being challenged.

You might have a tough exterior and a tenacious never-back-down attitude, but you’re a rock-solid friend who’ll do anything for the people you care about.'''

pink_type = '''Soaring above it all, Winged Pikmin don’t get bogged down in the details, and neither do you. You’re a team player to your core, eager to help anyone who needs it. Determined to do it all, you feel like you’re flying when you spot an opportunity to swoop in and help out, and like a Pikmin, you know there’s you’re stronger when everyone works together.

Far from having your head in the clouds, you’re reaching for the sky with your lofty goals and unstoppable ambition. Don’t worry, we know you’ll land amongst the stars!'''

white_type = '''White Pikmin are poisonous, so if you’re feeling a little rough around the edges lately, these small but mighty creatures can totally relate. Just because you’re going through it doesn’t mean you can’t handle it. In fact, you’re more courageous than you think!

These Pikmin can handle toxic environments better than most – and so can you! Navigating everything from bad days to friend fights seems to be your strong suit. You’re introspective and in tune with your feelings, and we’re betting you give seriously great advice.'''

pikmin_types = [red_type, blue_type, yellow_type, ice_type, purple_type, rock_type, pink_type, white_type]


# %% [markdown]
# __MBTI 16personalities scraping__

# %%
personalities = ['intj', 'intp', 'entj', 'entp', 'infp', 'infj', 'enfj', 'enfp', 'istj', 'isfj', 'estj', 'esfj', 'istp', 'isfp', 'estp', 'esfp']

mbti_df = pd.DataFrame({
    'personality': [],
    'red': [], 
    'blue': [],
    'yellow': [],
    'ice': [],
    'purple': [],
    'rock': [],
    'pink': [],
    'white': []
})
for perso in personalities:
    data = [perso]

    url = 'https://www.16personalities.com/' + perso + '-personality'
    response = requests.get(url)
    clean_response = bs4.BeautifulSoup(response.text, 'html.parser')
    
    p = clean_response.find_all('p')
    blurb = str(p[5].text)

    vectorizer = TfidfVectorizer(stop_words='english')
    
    for pik in pikmin_types:
        tfidf_matrix = vectorizer.fit_transform([pik, blurb])
        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        data.append(float(cosine_sim[0][0]))

    mbti_df.loc[len(mbti_df)] = data

mbti_df


# %%
for col in mbti_df.columns[1:]:
    col_list = list(mbti_df[col])
    mbti_list = list(mbti_df['personality'])

    mean = statistics.mean(col_list)
    col_list_sorted = sorted(col_list, reverse=True)

    pers_list = []
    for num in col_list_sorted:
        i = col_list.index(num)
        pers = mbti_list[i]
        pers_list.append(pers)
    
    print(col, ':', mean, pers_list)

# %%
colors = ['red', 'blue', 'yellow', 'ice', 'purple', 'rock', 'pink', 'white']

matching_df = pd.DataFrame({
    'mbti': [],
    'score': [],
    'color': []
})
for i in range(len(mbti_df)):
    data = []

    row = list(mbti_df.loc[i])
    data.append(row[0])

    j = 1
    highest_score = 0
    best_color = 'none'
    for col in colors:
        score = row[j]
        if score > highest_score:
            highest_score = score
            best_color = col

        j += 1
    
    data.append(highest_score)
    data.append(best_color)

    matching_df.loc[len(matching_df)] = data
    
matching_df



