import pandas as pd 
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
#from eda import load_data

def text_line(line):
    '''Takes in a line of html from the description column
    and converts to text.'''    
    soup = BeautifulSoup(line, 'html.parser')
    text = soup.text
    text2 = text.replace(u'\xa0', u' ')
    return text2

def text_work(df, max_features = 100):
    '''Removes the description column, applies the text_line function
    through a lambda  function, and returns a dense matrix of the text features.
    Concats the text features to the original dataframe.'''
    descriptions = df.pop('description')
    text_descriptions = descriptions.map(lambda line: text_line(line))
    t = TfidfVectorizer(max_features = max_features)
    denseText = t.fit_transform(text_descriptions)
    denseText = denseText.todense()
    d = pd.DataFrame(denseText)
    df2 = pd.concat([df, d], axis=1)
    return df2

def get_num_caps(df):
    '''Removes the description column, applies the text_line function
    through a lambda  function, and returns a dense matrix of the text features.
    Concats the text features to the original dataframe.'''
    descriptions = df.pop('description')
    text_descriptions = descriptions.map(lambda line: text_line(line))
    num_caps_freq = text_descriptions.map(lambda text: sum(1 for c in text if c.isupper())/(float(len(text)) if len(text) !=0 else 1))
    df['num_caps_freq'] = num_caps_freq
    return df

# if __name__ == '__main__':
    #df = load_data()
    # df2 = text_work(df)