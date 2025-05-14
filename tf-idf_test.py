# import json
# import math
# from sklearn.feature_extraction.text import TfidfVectorizer

# def cosine(u, v):
#     dot = sum(a*b for a, b in zip(u, v))
#     na = math.sqrt(sum(a*a for a in u))
#     nb = math.sqrt(sum(b*b for b in v))
#     return dot/(na*nb) if na and nb else 0.0

# def recommend_tfidf_user(data, attrs, user):
#     names = list(data.keys())
#     docs  = list(data.values())
#     vectorizer   = TfidfVectorizer(stop_words="english")
#     tfidf_matrix = vectorizer.fit_transform(docs)
#     idf_lookup   = dict(zip(vectorizer.get_feature_names_out(), vectorizer.idf_))
#     vocab        = vectorizer.vocabulary_

#     # build the user profile vector in TF–IDF space
#     D = len(vocab)
#     profile = [0.0] * D
#     for attr, score in zip(attrs, user):
#         idx = vocab.get(attr.lower())
#         if idx is not None:
#             profile[idx] = score * idf_lookup[attr.lower()]

#     # compute similarity against every item
#     sims = []
#     for i, name in enumerate(names):
#         item_vec = tfidf_matrix[i].toarray()[0].tolist()
#         sims.append((name, cosine(profile, item_vec)))

#     # return top 3
#     sims = sorted(sims, key=lambda x: x[1], reverse=True)[:3]
#     return sims

# if __name__ == "__main__":
#     movies = {
#         "The Lunar Heist":  "A ragtag crew of outlaws plans a high-stakes robbery on a corporate lunar mining colony, navigating zero-gravity security and moral dilemmas.",
#         "Echoes of Tomorrow":  "A time-traveling detective must solve crimes that erase their own past, racing paradoxes and battling a secret organization across centuries.",
#         "Whispering Pines":  "In an isolated mountain town, residents uncover hidden messages in the forest’s whispers that point to a decades-old unsolved disappearance.",
#         "Cybernetic Dawn":  "In a neon-soaked megacity ruled by A.I., a lone hacker and a disillusioned android join forces to spark a revolution for human freedom.",
#         "The Last Alchemist":  "A young scholar travels ancient lands in search of the Philosopher’s Stone, facing alchemical trials, betrayal, and the price of immortality.",
#         "Shadows of Eden":  "A fallen angel hiding among dystopian skyscrapers seeks redemption by protecting a child prophesied to save both Heaven and Earth.",
#         "Crimson Skies":  "Ace pilots duel in massive skyships above a war-torn empire, forging unlikely alliances to thwart a tyrant’s plan to dominate the skies.",
#         "Voyage to Avalon":  "A disillusioned knight sails into misty seas to find the legendary island of Avalon, confronting mythical beasts and his own haunted past.",
#         "Silent Symphony":  "A deaf composer struggles to complete his final symphony, finding new ways to “hear” music through vibrations and the hearts of those around them.",
#         "Neon Samurai":  "A ronin samurai with a cybernetic arm battles corrupt corporate shoguns in a rain-drenched, neon-lit dystopia where honor meets technology.",
#         "Winter’s Embrace":  "Two strangers stranded in an Alaskan blizzard forge a fragile bond as they seek shelter in an abandoned cabin, confronting loss and hope.",
#         "Forgotten Legends":  "An archaeologist discovers remnants of a civilization erased from history and must decode its secrets before a rival faction buries the truth.",
#         "Quantum Rift":  "When a physics experiment rips open a portal to alternate realities, a team of scientists must navigate shifting laws of nature to survive.",
#         "The Painted Veil":  "A tormented painter’s portraits begin predicting future tragedies, forcing her to choose between artistic truth and the safety of loved ones.",
#         "Midnight Carnival":  "Performers at a traveling carnival awaken supernatural forces under the full moon, turning sideshows into scenes of wonder and terror.",
#         "Garden of Whispers":  "A botanist discovers a hidden garden where plants communicate telepathically, unraveling conspiracies that could alter humanity’s future.",
#         "Iron Frontier":  "Settlers on Mars’s harsh frontier clash over water rights and survival, igniting a rebellion that will determine the colony’s destiny.",
#         "Echo Beach":  "Old friends reunite at a coastal town to restore a decaying beach house and confront the echoes of their shared—and fractured—past.",
#         "Soul Traders":  "In a black-market bazaar, mercenaries barter human souls for power, and one rogue agent seeks to free the trapped spirits before it’s too late.",
#         "Dreamweaver":  "A dream-hacker dives into the subconscious of a comatose friend, facing surreal nightmares and hidden memories to bring them back to life."
#     }

#     test_cases = [
#         (["heist", "space", "action"], [5, 4, 3]),
#         (["detective", "time-travel", "mystery"], [5, 4, 3]),
#         (["forest", "horror", "suspense"], [5, 4, 2]),
#         (["cyberpunk", "ai", "rebellion"], [5, 3, 2]),
#         (["alchemy", "fantasy", "adventure"], [4, 5, 3]),
#         (["angel", "redemption", "dystopia"], [4, 3, 5]),
#         (["skyship", "war", "empire"], [5, 4, 2]),
#         (["kingdom", "quest", "myth"], [4, 5, 3]),
#         (["music", "drama", "inspiration"], [3, 5, 4]),
#         (["dream", "subconscious", "thriller"], [4, 3, 5]),
#         (["gritty", "crime", "thriller"], [5, 4, 3]),
#         (["dark", "crime", "thriller"], [5, 4, 3]),
#         (["sci-fi", "futuristic", "tech"], [5, 4, 3]),
#         (["science-fiction", "futuristic", "tech"], [5, 4, 3]),
#         (["heart-warming", "romance", "family"], [5, 4, 3]),
#         (["feel-good", "romance", "family"], [5, 4, 3]),
#         (["funny", "comedy", "witty"], [5, 4, 3]),
#         (["humorous", "comedy", "witty"], [5, 4, 3]),
#         (["epic", "fantasy", "adventure"], [5, 4, 3]),
#         (["grand", "fantasy", "adventure"], [5, 4, 3]),
#     ]

#     # Open the output file for writing all test results
#     with open("ttf_tests.txt", "w") as out:
#         for attrs, user in test_cases:
#             recs = recommend_tfidf_user(movies, attrs, user)
#             # format as "Movie score, Movie score, Movie score"
#             line = ", ".join(f"{name} {score:.3f}" for name, score in recs)
#             out.write(line + "\n")




import json
import math
from sklearn.feature_extraction.text import TfidfVectorizer

def cosine(u, v):
    dot = sum(a*b for a, b in zip(u, v))
    na = math.sqrt(sum(a*a for a in u))
    nb = math.sqrt(sum(b*b for b in v))
    return dot/(na*nb) if na and nb else 0.0

def recommend_tfidf_user(data, attrs, user):
    names = list(data.keys())
    docs  = list(data.values())
    vectorizer   = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(docs)
    idf_lookup   = dict(zip(vectorizer.get_feature_names_out(), vectorizer.idf_))
    vocab        = vectorizer.vocabulary_

    # build the user profile vector in TF–IDF space
    D = len(vocab)
    profile = [0.0] * D
    for attr, score in zip(attrs, user):
        idx = vocab.get(attr.lower())
        if idx is not None:
            profile[idx] = score * idf_lookup[attr.lower()]

    # compute similarity against every item
    sims = []
    for i, name in enumerate(names):
        item_vec = tfidf_matrix[i].toarray()[0].tolist()
        sims.append((name, cosine(profile, item_vec)))

    # return top 3
    sims = sorted(sims, key=lambda x: x[1], reverse=True)[:3]
    return sims

if __name__ == "__main__":
    books = {
        "The Clockwork Archivist": "A reclusive librarian discovers a sentient brass index that rearranges recorded history, forcing her to decide which memories humanity may keep—or forget forever.",
        "Paper Kingdoms": "In a realm where origami castles drift on ink-black seas, a pacifist scribe must chronicle an approaching war without letting his words ignite reality itself.",
        "The Silent Cartographer": "A mute mapmaker charts a continent that appears only in dreams, while rival explorers invade sleepers’ minds to steal its hidden resources.",
        "Ashes of the Oracle": "An exiled priestess consumes prophecies burned by heretics, tasting fractured futures as cosmic librarians hunt her to erase forbidden timelines.",
        "The Glass Apothecary": "A scandal-tarnished alchemist distills emotions into crystal vials, but a mysterious client’s order for bottled guilt threatens every soul in the city.",
        "Hollow Crown": "A farm girl inherits a kingdom whose throne devours rulers’ memories; she must govern wisely before her own identity erodes page by page.",
        "Song of Rust": "On a desert planet of iron dunes, nomad musicians carve melodies into scrap-metal bones to summon long-buried storm spirits and rival clans.",
        "Ink & Bone Dust": "Two thieving scribes race to find a quill carved from dragon bone—said to rewrite the curses stitched into centuries-old family grimoires.",
        "Weepers of the Tide": "Citizens of a drowned metropolis cultivate living pearls grown from their tears, sparking trade wars and forbidden love with surface dwellers.",
        "Gilded Shadows": "By day a debutante, by night an illusionist thief, a young woman steals corrupted dreams from aristocrats to ransom back their virtue.",
        "Starlight Ciphers": "An autistic codebreaker decodes alien constellations that foretell political assassinations on Earth, outwitting spies who doubt her gift.",
        "The Seventh Door": "Every seventh reflection in an antique mirror opens to a winter forest where time bargains in secrets; one orphan keeps stepping through.",
        "Ember & Epitaph": "A retired dragon-slayer now writes eulogies for monsters, uncovering a plot to resurrect the very beasts he once killed.",
        "Violet Harvest": "In a color-starved dictatorship, a botanist breeds illegal purple flowers that induce memories—making her the regime’s most wanted fugitive.",
        "Clocktower Orphans": "Street urchins living inside a colossal timepiece discover gears that rewind local moments, turning petty theft into temporal rebellion.",
        "Marble Saints": "A sculptor frees living statues bound by ancient vows, chiseling away their stone prisons while the church brands her an iconoclast.",
        "Relic of the Fallen Sky": "An archaeologist unearths a sky-shard that warps gravity, pitting floating rebels against earthbound empires for control of the air.",
        "The Bone Court": "In a tribunal of dragon skulls that judge truth by scent, an anosmic advocate must fake evidence to win justice for the innocent.",
        "Lanterns of Nowhere": "Wanderers in an endless night follow lanterns tethered to storytellers’ hearts; when a lantern dims, its story—and carrier—begins to fade.",
        "Frostbitten Psalms": "Monastic scribes etch prayers onto ice tablets that melt into reality; one novice spells a plague and must rewrite herself to undo it."
    }

    test_cases = [
        (["heist", "space", "action"], [5, 4, 3]),
        (["detective", "time-travel", "mystery"], [5, 4, 3]),
        (["forest", "horror", "suspense"], [5, 4, 2]),
        (["cyberpunk", "ai", "rebellion"], [5, 3, 2]),
        (["alchemy", "fantasy", "adventure"], [4, 5, 3]),
        (["angel", "redemption", "dystopia"], [4, 3, 5]),
        (["skyship", "war", "empire"], [5, 4, 2]),
        (["kingdom", "quest", "myth"], [4, 5, 3]),
        (["music", "drama", "inspiration"], [3, 5, 4]),
        (["dream", "subconscious", "thriller"], [4, 3, 5]),
        (["gritty", "crime", "thriller"], [5, 4, 3]),
        (["dark", "crime", "thriller"], [5, 4, 3]),
        (["sci-fi", "futuristic", "tech"], [5, 4, 3]),
        (["science-fiction", "futuristic", "tech"], [5, 4, 3]),
        (["heart-warming", "romance", "family"], [5, 4, 3]),
        (["feel-good", "romance", "family"], [5, 4, 3]),
        (["funny", "comedy", "witty"], [5, 4, 3]),
        (["humorous", "comedy", "witty"], [5, 4, 3]),
        (["epic", "fantasy", "adventure"], [5, 4, 3]),
        (["grand", "fantasy", "adventure"], [5, 4, 3]),
    ]

    # Write all test results to ttf_tests_books.txt
    with open("ttf_tests_books.txt", "w") as out:
        for attrs, user in test_cases:
            recs = recommend_tfidf_user(books, attrs, user)
            line = ", ".join(f"{name} {score:.3f}" for name, score in recs)
            out.write(line + "\n")
