
# Naive Bayes - Identifying the probable author of a message from it's previous texts

Spam filters basically keep statistics of which words were seen in messages that were classified as ham or spam. For this notebook, I was curious: at first, could I compile statistics on the usage of words by few peoples, then with these statistics in hand, identify the probable author of a snipet of text?

Intuition: If we know how Obama or Trump normally write, can we take a tweet, and from how it is written, identify who wrote it?

The success of this exercise highly depends on the quality of the text used as sources. However, it's an interesting toy project. 

Because of the quantity of transcripts available, I chose to use texts from Obama and Trump.

Sources:
* Obama: few first speeches from http://obamaspeeches.com/ 
* Trump: https://www.tampabay.com/florida-politics/buzz/2018/08/01/heres-a-full-transcript-of-president-trumps-speech-from-his-tampa-rally/ and https://www.politico.com/story/2018/09/25/trump-un-speech-2018-full-text-transcript-840043

And for the snipet of text, I used twitter:
* tweet 1: https://twitter.com/BarackObama/status/1044690296917962754
* tweet 2: https://twitter.com/realDonaldTrump/status/1045444544068812800
* tweet 3: https://twitter.com/realDonaldTrump/status/1045003711104331776
* tweet 4: https://twitter.com/BarackObama/status/1039512025406349312
* tweet 5: https://twitter.com/BarackObama/status/1034892109868945409
* tweet 6: https://twitter.com/realDonaldTrump/status/1043966388182953984

**Spoiler**: Using these 6 examples, we have a 5/6 success rate, which isn't perfect, but still, it's a good start. The first tweet comes out negative, under 60% for both.


```python
import pandas as pd
import numpy as np
import copy
import re
```


```python
%load_ext autoreload
%autoreload 2

# I am reusing the same update_probability function used in my previous naive bayes notebooks
from naive_bayes import update_probability
```

## Loading the source texts

This function loads the files containing the text from each authors. The shape of the file doesn't matter too much since we will flat it out in a dataframe that contains "author" and "word" at first. Later on, we add the statistics to it.


```python
def load_data(files):
    df = pd.DataFrame(columns=['author', 'word'])
    
    for author, file in files.items():
        data = pd.read_csv(file, names=['text'], delimiter='\n')
        data['author'] = author
        data['text'] = data['text'].str.lower()
        data['text'] = data['text'].str.replace('[^A-Za-z0-9\@\#\']', ' ')
        data['text'] = data['text'].str.replace('\s+', ' ')
        data['text'] = data['text'].str.strip()
        data['text'] = data['text'].str.split()
        for ll in data['text']:
            for ww in ll:
                df = df.append({'author': author, 'word': ww}, ignore_index=True)
    return df
```

##  Cleaning the lines from unuseful characters

This clean_line function looks a lot like the cleaning part of the load_data() function, however, because we are playing with pure text here instead of dataframes, I separated the part for the loading and cleaning. 

TODO: see if I can reuse something like this function for the load_data() function, instead of duplicating commands


```python
def clean_line(line):

    line = line.lower()
    line = re.sub(r'[^\w\s\'\@\#]',' ',line)
    line = re.sub(r'\s+', ' ', line)
    line = line.strip()
    line = line.split(" ")
    return line

def clean_lines(lines):
    newlines = []
    for l in lines:
        newlines.append(clean_line(l))
    return newlines
```

## Loading the training data

This is where we load the training data and we assign the content to its owner. We can easily add more authors if desired, as opposed to traditional spam filters that only discriminate between good and bad text.


```python
files = {
    'obama': 'training_obama.txt', 
    'trump': 'training_trump.txt'
}


df = load_data(files)
df.describe()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>author</th>
      <th>word</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>28647</td>
      <td>28647</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>2</td>
      <td>3374</td>
    </tr>
    <tr>
      <th>top</th>
      <td>obama</td>
      <td>the</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>17980</td>
      <td>1280</td>
    </tr>
  </tbody>
</table>
</div>



## Adding the statistics 

This function calculates the following:
* How many time an author used a word, devided by everyone who used that word (true positives)
* How many time others used that same word, devided by everyone who used that word (false positives)
* True negatives = 1 - true positives
* False negatives = 1 - false positives

In addition, we keep the count of each word per author and for the others, which is useful for filtering later on.


```python
def add_statistics(df):
    metrics = pd.DataFrame(columns=['author', 'word', 'c_author','c_others', 'tp', 'tn', 'fp', 'fn'])

    for index, row in df.iterrows():
        if metrics.loc[(metrics['author'] == row['author'])& (metrics['word'] == row['word'])].empty:
            count_word_for_author = len(df.loc[(df['word'] == row['word']) & (df['author'] == row['author'])].index)+1
            count_word_for_others = len(df.loc[(df['word'] == row['word']) & (df['author'] != row['author'])].index)+1

            metrics = metrics.append({
                'author': row['author'],
                'word': row['word'],
                'c_author': count_word_for_author,
                'c_others': count_word_for_others,
                'tp': count_word_for_author/(count_word_for_author + count_word_for_others),
                'tn': 1 - (count_word_for_author/(count_word_for_author + count_word_for_others)),
                'fp': count_word_for_others/(count_word_for_author + count_word_for_others),
                'fn': 1 - (count_word_for_others/(count_word_for_author + count_word_for_others))
            }, ignore_index=True)
    
    return metrics
```


```python
metrics_unfiltered = add_statistics(df)
print(metrics_unfiltered.sort_values(by=['c_author', 'c_others'], ascending=False))
```

         author           word c_author c_others        tp        tn        fp  \
    9     obama            the      828      454  0.645866  0.354134  0.354134   
    35    obama            and      756      391  0.659111  0.340889  0.340889   
    30    obama             to      536      311  0.632822  0.367178  0.367178   
    20    obama             of      471      230  0.671897  0.328103  0.328103   
    2353  trump            the      454      828  0.354134  0.645866  0.645866   
    101   obama           that      410      118  0.776515  0.223485  0.223485   
    2368  trump            and      391      756  0.340889  0.659111  0.659111   
    108   obama              a      352      163  0.683495  0.316505  0.316505   
    87    obama             we      316      216  0.593985  0.406015  0.406015   
    2344  trump             to      311      536  0.367178  0.632822  0.632822   
    83    obama             in      272      176  0.607143  0.392857  0.392857   
    23    obama            our      253      138  0.647059  0.352941  0.352941   
    3     obama              i      238      160  0.597990  0.402010  0.402010   
    2360  trump             of      230      471  0.328103  0.671897  0.671897   
    14    obama            for      225      114  0.663717  0.336283  0.336283   
    2351  trump             we      216      316  0.406015  0.593985  0.593985   
    64    obama             is      190      104  0.646259  0.353741  0.353741   
    41    obama           this      187       71  0.724806  0.275194  0.275194   
    2346  trump             in      176      272  0.392857  0.607143  0.607143   
    16    obama            you      168      154  0.521739  0.478261  0.478261   
    2356  trump              a      163      352  0.316505  0.683495  0.683495   
    2341  trump              i      160      238  0.402010  0.597990  0.597990   
    2386  trump            you      154      168  0.478261  0.521739  0.521739   
    259   obama            who      151       21  0.877907  0.122093  0.122093   
    96    obama             it      139      118  0.540856  0.459144  0.459144   
    2366  trump            our      138      253  0.352941  0.647059  0.647059   
    102   obama            are      136       79  0.632558  0.367442  0.367442   
    86    obama            but      136       59  0.697436  0.302564  0.302564   
    17    obama           have      126       99  0.560000  0.440000  0.440000   
    183   obama           will      120       54  0.689655  0.310345  0.310345   
    ...     ...            ...      ...      ...       ...       ...       ...   
    4259  trump          prize        2        1  0.666667  0.333333  0.333333   
    4260  trump       sustains        2        1  0.666667  0.333333  0.333333   
    4262  trump           deep        2        1  0.666667  0.333333  0.333333   
    4266  trump       treasure        2        1  0.666667  0.333333  0.333333   
    4269  trump        chamber        2        1  0.666667  0.333333  0.333333   
    4270  trump      listening        2        1  0.666667  0.333333  0.333333   
    4272  trump        patriot        2        1  0.666667  0.333333  0.333333   
    4274  trump        intense        2        1  0.666667  0.333333  0.333333   
    4276  trump       homeland        2        1  0.666667  0.333333  0.333333   
    4279  trump          souls        2        1  0.666667  0.333333  0.333333   
    4284  trump     scientific        2        1  0.666667  0.333333  0.333333   
    4285  trump  breakthroughs        2        1  0.666667  0.333333  0.333333   
    4287  trump            art        2        1  0.666667  0.333333  0.333333   
    4289  trump          erase        2        1  0.666667  0.333333  0.333333   
    4290  trump           draw        2        1  0.666667  0.333333  0.333333   
    4291  trump        ancient        2        1  0.666667  0.333333  0.333333   
    4292  trump         wisdom        2        1  0.666667  0.333333  0.333333   
    4294  trump        regions        2        1  0.666667  0.333333  0.333333   
    4295  trump        unleash        2        1  0.666667  0.333333  0.333333   
    4296  trump    foundations        2        1  0.666667  0.333333  0.333333   
    4298  trump        vehicle        2        1  0.666667  0.333333  0.333333   
    4299  trump       survived        2        1  0.666667  0.333333  0.333333   
    4301  trump      prospered        2        1  0.666667  0.333333  0.333333   
    4302  trump      cherished        2        1  0.666667  0.333333  0.333333   
    4303  trump      unfolding        2        1  0.666667  0.333333  0.333333   
    4304  trump    peacemaking        2        1  0.666667  0.333333  0.333333   
    4307  trump        resolve        2        1  0.666667  0.333333  0.333333   
    4308  trump    flourishing        2        1  0.666667  0.333333  0.333333   
    4311  trump       thankful        2        1  0.666667  0.333333  0.333333   
    4314  trump          glory        2        1  0.666667  0.333333  0.333333   
    
                fn  
    9     0.645866  
    35    0.659111  
    30    0.632822  
    20    0.671897  
    2353  0.354134  
    101   0.776515  
    2368  0.340889  
    108   0.683495  
    87    0.593985  
    2344  0.367178  
    83    0.607143  
    23    0.647059  
    3     0.597990  
    2360  0.328103  
    14    0.663717  
    2351  0.406015  
    64    0.646259  
    41    0.724806  
    2346  0.392857  
    16    0.521739  
    2356  0.316505  
    2341  0.402010  
    2386  0.478261  
    259   0.877907  
    96    0.540856  
    2366  0.352941  
    102   0.632558  
    86    0.697436  
    17    0.560000  
    183   0.689655  
    ...        ...  
    4259  0.666667  
    4260  0.666667  
    4262  0.666667  
    4266  0.666667  
    4269  0.666667  
    4270  0.666667  
    4272  0.666667  
    4274  0.666667  
    4276  0.666667  
    4279  0.666667  
    4284  0.666667  
    4285  0.666667  
    4287  0.666667  
    4289  0.666667  
    4290  0.666667  
    4291  0.666667  
    4292  0.666667  
    4294  0.666667  
    4295  0.666667  
    4296  0.666667  
    4298  0.666667  
    4299  0.666667  
    4301  0.666667  
    4302  0.666667  
    4303  0.666667  
    4304  0.666667  
    4307  0.666667  
    4308  0.666667  
    4311  0.666667  
    4314  0.666667  
    
    [4315 rows x 8 columns]


## Setting the priors

We have two sources, therefore, each source should have a prior of 50%


```python
prior = {}
users = sorted(metrics_unfiltered['author'].unique())
for u in users:
    prior[u] = 1 / len(metrics_unfiltered['author'].unique())
    print("{}: {:.2f}".format(u, prior[u]))
```

    obama: 0.50
    trump: 0.50


## Analysing text snipets

The analyse_texts() function converts the metrics dataframe in a format that the update_probability() function can handle. 

The function will go through each word of the text snipet, check if there is an associated probability to it (only probabilities of the target author are received in parameter), and if a word isn't found, which means that that author never used that word before, we assign a probability 50% to it, basically not altering the posterior probability.

Experimentation notes: I did perform some test by setting a 49% probability instead of 50% for never used words, which added a negative bias for new words, but I removed it because I also added a filtering on the words to keep only the bottom 80%, removing over repeated English words.


```python
def analyse_texts(prior, texts, probabilities_author, probabilities_others, debug):
    posterior = prior
    tests = {}

    for tt in texts:
        prob = probabilities_author.loc[probabilities_author['word'] == tt]
        prob_others = probabilities_others.loc[probabilities_others['word'] == tt]

        if tt not in tests.keys():
            if not prob.empty: # if the searched word was used by the current author
                tests[tt] = {
                    'True': {
                        'Positive': prob.head(1)['tp'].values[0],
                        'Negative': prob.head(1)['tn'].values[0]
                    },
                    'False': {
                        'Positive': prob.head(1)['fp'].values[0],
                        'Negative': prob.head(1)['fn'].values[0]
                    }
                }
            elif not prob_others.empty: # if the word was not used by the author, check for other authors
                tests[tt] = {
                    'True': {
                        'Positive': prob_others.head(1)['tp'].values[0],
                        'Negative': prob_others.head(1)['tn'].values[0]
                    },
                    'False': {
                        'Positive': prob_others.head(1)['fp'].values[0],
                        'Negative': prob_others.head(1)['fn'].values[0]
                    }
                }
            else: # if it's the first time we see a word, assign a probability of 50%
                tests[tt] = {
                    'True': {
                        'Positive': 0.5,
                        'Negative': 0.5
                    },
                    'False': {
                        'Positive': 0.5,
                        'Negative': 0.5
                    }
                }
        if not prob.empty:
            posterior = update_probability(posterior, tt, tests, 'True', debug)
        elif not prob_others.empty:
            posterior = update_probability(posterior, tt, tests, 'False', debug)
        else:
            posterior = update_probability(posterior, tt, tests, 'False', debug)

    return posterior
```


```python
def lookup_text(lines, prior, metrics):
    for ll in lines:
        print('Given the text "{}"...'.format(" ".join(ll)))
        for author in prior.keys():
            posterior = analyse_texts(prior[author], ll, metrics.loc[metrics['author'] == author], metrics.loc[metrics['author'] != author], False)
            print("\tProbability that {} wrote this text is: {:.3f}%".format(author, 100 * posterior))
        print("\n")
```

## Filtering metrics to keep only the buttom 80% used words.

I am removing over repeating words here because they are skewing the probablities in the direction of the author that uses the most articles...


```python
metrics = metrics_unfiltered[metrics_unfiltered.c_author < metrics_unfiltered.c_author.quantile(.8)]
print(metrics.sort_values(by=['c_author', 'c_others'], ascending=False).head())
print(len(metrics['word'].unique()))
```

         author        word c_author c_others        tp        tn        fp  \
    3705  trump      change        5       68  0.068493  0.931507  0.931507   
    2821  trump      cannot        5       31  0.138889  0.861111  0.861111   
    1468  obama       doing        5       26  0.161290  0.838710  0.838710   
    3266  trump  washington        5       26  0.161290  0.838710  0.838710   
    2342  trump          am        5       25  0.166667  0.833333  0.833333   
    
                fn  
    3705  0.068493  
    2821  0.138889  
    1468  0.161290  
    3266  0.161290  
    2342  0.166667  
    2955


## Querying our model using tweets

We are now ready to throw tweets at our model. The text comes from the following tweets:

* tweet 1: https://twitter.com/BarackObama/status/1044690296917962754
* tweet 2: https://twitter.com/realDonaldTrump/status/1045444544068812800
* tweet 3: https://twitter.com/realDonaldTrump/status/1045003711104331776
* tweet 4: https://twitter.com/BarackObama/status/1039512025406349312
* tweet 5: https://twitter.com/BarackObama/status/1034892109868945409
* tweet 6: https://twitter.com/realDonaldTrump/status/1043966388182953984

This function could also be used to analyse each sentenses from an anonymous op-ed against the known speeches of candidate writers, and evaluate if a full text is likely or not to have been written by multiple authors or from a single one.


```python
lines_of_text = ["The antidote to government by a powerful few is government by the organized, energized many. This National Voter Registration Day, make sure you're registered, vote early if you can, or show up on November 6. This moment is too important to sit out.",
                "Judge Kavanaugh showed America exactly why I nominated him. His testimony was powerful, honest, and riveting. Democrats’ search and destroy strategy is disgraceful and this process has been a total sham and effort to delay, obstruct, and resist. The Senate must vote!",
                "Congressman Lee Zeldin is doing a fantastic job in D.C. Tough and smart, he loves our Country and will always be there to do the right thing. He has my Complete and Total Endorsement!",
                "We will always remember everyone we lost on 9/11, thank the first responders who keep us safe, and honor all who defend our country and the ideals that bind us together. There's nothing our resilience and resolve can’t overcome, and no act of terror can ever change who we are.",
                "Yesterday I met with high school students on Chicago’s Southwest side who spent the summer learning to code some pretty cool apps. Michelle and I are proud to support programs that invest in local youth and we’re proud of these young people.",
                "Going to New York. Will be with Prime Minister Abe of Japan tonight, talking Military and Trade. We have done much to help Japan, would like to see more of a reciprocal relationship. It will all work out!"]

clean_lines_of_texts = clean_lines(lines_of_text)

posterior = lookup_text(clean_lines_of_texts, prior, metrics)
```

    Given the text "the antidote to government by a powerful few is government by the organized energized many this national voter registration day make sure you're registered vote early if you can or show up on november 6 this moment is too important to sit out"...
    	Probability that obama wrote this text is: 42.131%
    	Probability that trump wrote this text is: 57.869%
    
    
    Given the text "judge kavanaugh showed america exactly why i nominated him his testimony was powerful honest and riveting democrats search and destroy strategy is disgraceful and this process has been a total sham and effort to delay obstruct and resist the senate must vote"...
    	Probability that obama wrote this text is: 9.934%
    	Probability that trump wrote this text is: 90.066%
    
    
    Given the text "congressman lee zeldin is doing a fantastic job in d c tough and smart he loves our country and will always be there to do the right thing he has my complete and total endorsement"...
    	Probability that obama wrote this text is: 2.500%
    	Probability that trump wrote this text is: 97.500%
    
    
    Given the text "we will always remember everyone we lost on 9 11 thank the first responders who keep us safe and honor all who defend our country and the ideals that bind us together there's nothing our resilience and resolve can t overcome and no act of terror can ever change who we are"...
    	Probability that obama wrote this text is: 99.988%
    	Probability that trump wrote this text is: 0.012%
    
    
    Given the text "yesterday i met with high school students on chicago s southwest side who spent the summer learning to code some pretty cool apps michelle and i are proud to support programs that invest in local youth and we re proud of these young people"...
    	Probability that obama wrote this text is: 99.956%
    	Probability that trump wrote this text is: 0.044%
    
    
    Given the text "going to new york will be with prime minister abe of japan tonight talking military and trade we have done much to help japan would like to see more of a reciprocal relationship it will all work out"...
    	Probability that obama wrote this text is: 3.399%
    	Probability that trump wrote this text is: 96.601%
    
    


## How to improve?

This notebook uses Naive Bayes, which naively consider each word as independent events. This isn't true in reality, but allows us to run a simple function over all words without too much computer horse power. 

* Add more authors, which will give a better distribution of common words, and make the one less used stand out.
* The accuracy could most probably be improved by using markov chains. To do in a future notebook...
* An other approach would be through machine learning, whatever the model used... also for a future notebook.
