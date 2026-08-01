## Points Bot

This is a project for attempting to identify point-worthy cards in the 10-point Canadian Highlander (CHL) MtG format. 

### Why
- I thought it would be fun
- I wanted to see if the resulting model would accurate *at all* (my starting expectation was that this would be a total waste of time)
- In a world where a good enough model for card power exists, points decisions would have less room for human bias
- Magic is fun to think about.

### The story so far
- Was originally going to do 7pt because it's the cooler format, but tournament result postings are so sparse and unstructured it'd be too painful to gather enough data
- Training on simple metadata and results alone was **useless**
- Scraping tags is a nightmare; every day I pray for an API
- I am aware that edhrec is a Commander resource; it's still 100-card singleton that some weirdos play competitively - some of their data should be useful
- tag embeddings made a huge difference in model performance, hooray I can't wait to rescrape them and have it take a million years because I need to render the whole damn web page for each card
- Regressor attempts have been kind of a joke; not going to bother paying too much attention to them anymore. Nothing that's unpointed right now is likely to ever be worth more than 1 point anyway.
- Visualising pointed cards in 2d embedding space is neat
- Surprised to see the classifier is semi-coherent; Wood Elves being a suspect is pretty damn funny though.
- Putting all the oracle text through tf-idf might be a good idea.
- It was a good idea
- Would be nice to understand what the embedding features actually represent, reverse engineering them based on rank-order might help
- I dunno why I'm generating an embedding from the keywords field; this could just be multi-hot instead
- Actually, it's kind of a pointless feature given only 3 of the pointed cards have it
- Why am I not getting bigrams and trigrams when I do tf-idf???? idiot