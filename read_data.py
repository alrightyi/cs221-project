"""TODO(hoivan): DO NOT SUBMIT without one-line documentation for read_data.
read and preprocess review data from roger ebert
TODO(hoivan): DO NOT SUBMIT without a detailed description of read_data.
"""
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
            txt += re.sub('\'|\\|\/|\(|\)|\[|\]|\{|\}', '', detail)+('\n' if alt % 2 == 1 else ' ')
            alt += 1
        #for more_detail in self.more_details:
        #    txt += re.sub('\'|\\|\/|\(|\)|\[|\]|\{|\}', '', more_detail)+'\n'
        for body in self.body:
            if body == '\'Advertisement\'' or 'googletag.cmd.push' in body:
                continue
            txt += re.sub('\'|\\|\/|\(|\)|\[|\]|\{|\}', '', body)+'\n'
        txt += '\n'
        return txt

def loadReviews():
    f = open('store.pckl', 'rb')
    reviews = pickle.load(f)
    f.close()
    
    keys = list(reviews.keys())
    length = 200
    random.Random(4).shuffle(keys)
    txt = open('roger_ebert_200.txt', 'w+')
    for key in keys[0:length]:
        txt.write(repr(reviews[key]))
    txt.close()

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
