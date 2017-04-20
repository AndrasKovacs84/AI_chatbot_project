import nltk
import pprint


def user_input():
    entry = input("user input:")
    return entry


def main():
    entry = "This is an example sentence to tokenize. I wonder if it will work. It should, shouldn't it?"
    # entry = user_input()
    print(nltk.sent_tokenize(entry))
    print(nltk.word_tokenize(entry))
    pos_tagged = nltk.pos_tag(nltk.word_tokenize(entry))
    print(nltk.pos_tag(nltk.word_tokenize(entry)))
    chunkGram = """Chunk: {<RB.?>*<VB.?>*<NNP><NP>?}"""
    # grammar1 = nltk.data.load('file:mygrammar.cfg')
    grammar1 = nltk.CFG.fromstring("""
                                    S -> NP VP
                                    VP -> V NP | V NP PP
                                    PP -> P NP
                                    V -> "saw" | "ate" | "walked"
                                    NP -> "John" | "Mary" | "Bob" | Det N | Det N PP
                                    Det -> "a" | "an" | "the" | "my"
                                    N -> "man" | "dog" | "cat" | "telescope" | "park"
                                    P -> "in" | "on" | "by" | "with"
                                    """)
    grammar = nltk.CFG.fromstring("""
                                  S -> NP VP
                                  PP -> P NP
                                  NP -> Det N | NP PP
                                  VP -> V NP | VP PP
                                  Det -> 'a' | 'the'
                                  N -> 'dog' | 'cat'
                                  V -> 'chased' | 'sat'
                                  P -> 'on' | 'in'
                                  """)

    chunkParser = nltk.RegexpParser(chunkGram)
    sr_parser = nltk.ShiftReduceParser(grammar1)
    chart_parser = nltk.ChartParser(grammar)

    chunked = chunkParser.parse(pos_tagged)
    pos_to_draw = chart_parser.parse(pos_tagged)

    pos_to_draw.draw()
    # print(nltk.help.upenn_tagset())


if __name__ == '__main__':
    main()