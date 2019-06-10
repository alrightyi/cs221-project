'''
Read and preprocess text data from roger ebert
Method:
1) Filter list of movie reviews authored by Roger Ebert from 
https://www.rogerebert.com/reviews. Parse the ist using lxml and BeautifulSoup
2) For each movie review listing, get the movie metadata, actual rating and review 
text from https://www.rogerebert.com/reviews
3) Store the data in class Review and store locally using Pickle
4) Separate method to generate stats such as how many movie reviews in total, 
avg rating, max word length.
'''
from bs4 import  BeautifulSoup
import lxml
import requests
import pickle
import sys
import resource
import re
import random

CURL = "https://www.rogerebert.com/reviews?cmd=ChdjYS1wdWItNDY5ODU5NTUxOTc1NjkxMg&filters%5Bgreat_movies%5D%5B%5D=&filters%5Bno_stars%5D%5B%5D=&filters%5Bno_stars%5D%5B%5D=1&filters%5Btitle%5D=&filters%5Breviewers%5D%5B%5D=50cbacd5f3b43b53e9000003&filters%5Bgenres%5D=&sort%5Border%5D=oldest&page="

BASE_URL = "https://www.rogerebert.com"

CREW_TYPES = ["Written by", "Directed by", "Written and Directed by", "Produced by", "Photographed by", "Edited by", "Music by"]

class Review:
    def __init__(self, url=None, title=None, year=None):
        # URL to the review
        self.url = url

        # Movie title
        self.title = title

        # Release year
        self.year = year

        # Cast and Crew
        self.details = []
        
        # Rating
        self.rr = 0
        
        # Genres
        self.more_details = []
        
        # MPAA rating
        self.mr = None
        
        # Running Time
        self.rt = 0
        
        # review body
        self.body = []
        
    
    def __repr__(self):
        
        txt = ''
        alt = 0
        for detail in self.details:
            detail = detail.lower()
            detail = re.sub('[^a-zA-z0-9\s]', '', detail)
            txt += detail+('\n' if alt % 2 == 1 else ' ')
            alt += 1
        #for more_detail in self.more_details:
        #    txt += re.sub('\'|\\|\/|\(|\)|\[|\]|\{|\}', '', more_detail)+'\n'
        for body in self.body[-6:-1]:
            if body == '\'Advertisement\'' or 'googletag.cmd.push' in body:
                continue
            body = body.lower()
            body = re.sub('[^a-zA-z0-9\s]', '', body)
            txt += body+' '
        txt += '\n'
        return txt

def loadReviews():
    f = open('store.pckl', 'rb')
    reviews = pickle.load(f)
    f.close()
    
    keys = list(reviews.keys())
    length = 2000
    random.Random(4).shuffle(keys)
    txt = open('ebert_last5_2000.txt', 'w+')
    for key in keys[0:length]:
        if len(reviews[key].body) > 100:
            continue
        txt.write(repr(reviews[key]))
    txt.close()
    
def reviewStats():
    f = open('store.pckl', 'rb')
    reviews = pickle.load(f)
    f.close()
    
    keys = list(reviews.keys())
    valid_keys = 0
    total_words = []
    min_words = float('inf')
    max_words = 0
    thumbs_up = 0
    thumbs_down = 0
    print("Total number of keys: ", len(keys))
    for key in keys:
        if len(reviews[key].body) > 100:
            continue
            
        valid_keys += 1
        body = reviews[key].body
        txt = ''
        for b in body:
            if b == '\'Advertisement\'' or 'googletag.cmd.push' in b:
                continue
            b = b.lower()
            b = re.sub('[^a-zA-z0-9\s]', '', b)
            txt += b+' '
        txt += '\n'
        words = [w for w in txt.split(' ') if w.strip() != '' or w == '\n']
        if len(words) > max_words:
            max_words = len(words)
        if len(words) < min_words:
            min_words = len(words)
        
        for w in words:
            total_words.append(w)
            
        if reviews[key].rr >= 3:
            thumbs_up +=1
        else:
            thumbs_down +=1
    
    print("Total reviews: ", valid_keys)
    print("Corpus length: ", len(total_words))
    print("Avg. words per review: ", len(total_words) / valid_keys)
    print("Max words: ", max_words)
    print("Min words: ", min_words)
    print("Total unique words: ", len(set(total_words)))
    print("Thumbs-ups: ", thumbs_up)
    print("Thumbs-downs: ", thumbs_down)
            

def getSummaries(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    else :
        #print(resource.getrlimit(resource.RLIMIT_STACK))
        #print(sys.getrecursionlimit())

        max_rec = 0x10000
        # May segfault without this line. 0x100 is a guess at the size of each stack frame.
        resource.setrlimit(resource.RLIMIT_STACK, [0x100 * max_rec, resource.RLIM_INFINITY])
        sys.setrecursionlimit(max_rec)
    
        page = 1 
        reviews = {}
        while True:
            file = requests.get(CURL+str(page))
            soup = BeautifulSoup(file.content, features="lxml")
            links = soup.find_all('figure',{'class':'movie review'})
            if len(links) == 0:
                break
            for link in links:
                a = link.find('h5', {'class':'title'}).a
                key = a.get('href')
                title = a.string
                year = link.find('span', {'class':'release-year'})
                if year != None:
                    year = year.string.strip('()')

                reviews[key] = Review(key,title=title,year=year)
                
                # get the details
                detail_file = requests.get(BASE_URL+key)
                detail_soup = BeautifulSoup(detail_file.content, features="lxml")
                details = detail_soup.find('section', {'class':'details'})
                if details == None:
                    print("No details found for: " + key)
                else:
                    for string in details.stripped_strings:
                        reviews[key].details.append(repr(string))
                    #print(reviews[key].details)
                
                more_details = detail_soup.find('section', {'class':'more-details'})
                if more_details == None:
                    print("No more-details found for: " + key)
                else:
                    for string in more_details.stripped_strings:
                        reviews[key].more_details.append(repr(string))
                    print(reviews[key].more_details)
                
                #rating
                rating = detail_soup.find('meta', {'itemprop':'ratingValue'})
                if rating == None:
                    print("no rating")
                else:
                    reviews[key].rr = float(rating.get('content'))
                    print("rating: ", reviews[key].rr)
                    
                # review body
                body = detail_soup.find('div', {'itemprop':'reviewBody'})
                if body == None:
                    print("No review body found for: " + key)
                else:
                    for string in body.stripped_strings:
                        reviews[key].body.append(repr(string))
                    #print(reviews[key].body)
                
            page += 1
            print("page: ", page)
        print("total entries: ", len(reviews))
        f = open('store.pckl', 'wb')
        pickle.dump(reviews, f)
        f.close()
    return 0                     

if __name__ == '__main__':
    #getSummaries(sys.argv)
    #loadReviews()
    reviewStats()
