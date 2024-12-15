# pikmin
Personal project using web scraping, APIs, and NLTK on the subject of Pikmin

Goal: Determine 'best' pikmin based on three metrics
- Cultural impact of pikmin
    - Via # of Reddit posts on r/Pikmin and r/PikminBloomApp
- Public outlook on pikmin
    - Via average # of upvotes on Reddit posts
    - Via NLP processing of comments on these posts determining 'positive' feedback
- Viability of pikmin in game
    - Via stats from Pikmin 4 via Pikipedia

Secondary goal: Determine which pikmin each of my friends is most like
- From MBTI (self reported from friends)
    - Scrape 16personalities for top traits of each personality
    - Generate similarity score between traits of personality and adjectives describing pikmin
