import spacy


class ParserAnnotationExtractor:
    def __init__(self):
        self.core = spacy.load("en_core_web_lg")

    def extract(self, text):
        document = self.core(text.lower())

        verbs = [token.lemma_ for token in document if token.pos_ == "VERB"]
        nouns = [token.lemma_ for token in document if token.pos_ == "NOUN"]

        pairs = []

        for index, verb in enumerate(verbs):
            verb_token = [t for t in document if t.lemma_ == verb][0]

            if verb_token.dep_ not in ["ROOT", "amod", "conj"] or (verb_token.dep_ == "conj"
                                                                   and document[verb_token.i + 1].text[-3:] == "ing"):
                continue
            for noun in nouns:
                token_noun = [t for t in document if t.lemma_ == noun][0]
                if token_noun.dep_ == "pobj" and token_noun.head.pos_ == "ADP":
                    continue
                pairs.append(f"{verb}_{noun}")

        return pairs

    def extract2(self, text):
        doc = self.core(text)
        objects = []

        for token in doc:
            if token.pos_ in ["NOUN", "PROPN"]:
                if token.dep_ == "pobj" and token.head.pos_ == "ADP":
                    continue
                objects.append(token)

        verbs = []
        for token in doc:
            if token.pos_ == "VERB":
                if token.dep_ in ["ROOT", "amod"]:
                    verbs.append(token)

        expanded_objects = []
        for obj in objects:
            expanded_objects.append(obj)
            for child in obj.children:
                if child.dep_ == "conj" and child.pos_ in ["NOUN", "PROPN"]:
                    expanded_objects.append(child)

        pairs = []
        for verb in verbs:
            for obj in expanded_objects:
                pairs.append(f"{verb.lemma_}_{obj.lemma_}")

        return pairs


if __name__ == "__main__":
    # text = "Catch up the cup and put it on the table"
    # objects_list = ["cup", "table"]
    #
    # extractor = AnnotationExtractor()
    # print(extractor.extract_actions_objects(text, objects_list))

    examples = [
        "Cut onion and put on the table",
        "Cut onion and carrots",
        "Put and start frying potatoes",
        "Grill the tomatoes in a pan and then put them in a plate",
        "Add oil to a pan and spread it well so as to fry the bacon",
        "Cook bacon until crispy, then drain on paper towel",
        "Add a bit of Worcestershire sauce to mayonnaise and spread it over the bread",
        "Place a piece of lettuce as the first layer, place the tomatoes over it",
        "Sprinkle salt and pepper to taste",
        "Place the bacon at the top",
        "Place a piece of bread at the top"
    ]

    extractor = ParserAnnotationExtractor()
    for ex in examples:
        print('1st:', ex, "->", extractor.extract(ex))
        print('2nd:', ex, "->", extractor.extract2(ex))
