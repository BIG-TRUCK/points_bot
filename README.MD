## Points Bot

This is a project for attempting to identify point-worthy cards in the 10-point Canadian Highlander (CHL) MtG format. 

### The problem

An active metagame contains thousands of cards, the majority of which do not warrant a second thought - narrowing the scope of cards to be reviewed would be useful for prompting more targeted discussion amongst the rules committee. Using a model to do it reduces the "squeaky wheel" effect where a small number of players complaining frequently about particular cards results in a disproportionate response. It could also solve the inverse, where genuinely high-powered and over-represented cards are ignored because a vocal minority of players with a vested interest in keeping those cards un/lower-pointed shout down genuine discussion.

- Explainability is not a huge concern, there will always be human assessment involved as a final step. The goal is for triage moreso than outright decision-making.
- "Suspect flagging" is the main goal; determining a specific points value is just something to tinker with, due to the heavy skew and low number of labelled cases it's unlikely any regressor models will perform well. (No surprises; an MAE of 1 where the predictions are ranging from 1-3 is trash)


### Notes
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
- At some point I'd like to remove the tournament result features and see what happens. It would be nice to be able to flag "sleeper" cards that could be useful in the format but have yet to appear in any tournaments.
- I should try multi-hot encoding some of the tags that show up in the feature importance charts under their embedding dimension - might also be nice to list particular *cards* as well as tags.