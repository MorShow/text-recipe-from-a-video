import ollama


class TagsToRecipeTransformer:
    def __init__(self):
        self.system_prompt = {
            'role': 'system',
            'content': 'You have a triplet '
                       '(action: <imperative_verb_describing_action> | '
                       'noun: <noun_describing_object_affected_by_action> |'
                       'target: <noun_describing_place_or_null>).'
                       'You should return a fragment from some food recipe.'
                       'It could be 1-3 sentences. Please, don`t add any additional information.'
                       'The answer MUST contain the info only about the things mentioned in the input,'
                       'but with the beautiful description.'
        }
        self.model = 'gemma3:4b-it-qat'
        self.temperature = 0.3

    def create_recipe(self, text: str | dict) -> str:
        if isinstance(text, dict):
            text = (f'action: {text.get("action", "null")} | '
                    f'noun: {text.get("noun", "null")} | '
                    f'target: {text.get("target", "null")}')

        response = ollama.chat(
            model=self.model,
            messages=[
                self.system_prompt,
                {'role': 'user', 'content': text}
            ],
            options={'temperature': self.temperature}
        )
        text = response['message']['content'].strip()
        return text


if __name__ == '__main__':
    t = TagsToRecipeTransformer()
    print(t.create_recipe('action: cut | noun: onion | target: bowl'))
    print(t.create_recipe({'action': 'cut', 'noun': 'onion', 'target': 'bowl'}))
