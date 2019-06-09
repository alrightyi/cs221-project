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
    avg_chars = 0
    avg_lines = 0
    avg_chars_per_line = 0
    avg_rating = 0
    min_chars = float('inf')
    max_chars = 0
    min_lines = float('inf')
    max_lines = 0
    avg_rating = 0
    print("Total number of keys: ", len(keys))
    for key in keys:
        body = reviews[key].body
        lines = len(body)
        avg_lines += lines
        if lines > max_lines:
            max_lines = lines 
            max_key = key
        if lines < min_lines:
            min_lines = lines 
            min_key = key
        rating = reviews[key].rr
        avg_rating += rating
        for line in body:
            chars = len(line)
            avg_chars += chars
            max_chars = chars if chars > max_chars else max_chars
            min_chars = chars if lines < min_chars else min_chars
    avg_lines /= len(keys)
    avg_chars /= len(keys)
    avg_chars_per_line = avg_chars / avg_lines
    avg_rating /= len(keys)
    
    print("Avg. line per review: ", avg_lines)
    print("Max lines: ", max_lines)
    print("Min lines: ", min_lines)
    print("Avg. chars per review: ", avg_chars)
    print("Max chars: ", max_chars)
    print("Min chars: ", min_chars)
    print("Avg. chars per line: ", avg_chars_per_line)
    print("Avg. rating: ", avg_rating)
    
    print("Max lines review: ", reviews[max_key])
    print("Min lines review: ", reviews[min_key])
    print("Max lines rating: ", reviews[max_key].rr)
    print("Min lines rating: ", reviews[min_key].rr)
            

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
    loadReviews()
    #reviewStats()
