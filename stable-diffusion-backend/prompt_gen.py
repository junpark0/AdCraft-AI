from expansion import FooocusExpansion
import random

expansion = FooocusExpansion()

class PromptGenerator:
    def __init__(self, themes, user_prompt):
        """
        Initialize the PromptGenerator with a dictionary of themes.
        :param themes: A dictionary where keys are theme names and values are booleans.
        """
        self.user_prompt = user_prompt
        self.themes = themes
        self.positive_prompts = []
        self.negative_prompts = []
        self.generate_prompts()

    def generate_prompts(self):
        """
        Generate positive and negative prompts based on the selected themes.
        """
        for theme, include in self.themes.items():
            if include:
                self.add_theme_prompts(theme)

    def add_theme_prompts(self, theme):
        """
        Add prompts specific to a theme.
        :param theme: The theme to add prompts for.
        """
        if theme == 'fantasy':
            self.positive_prompts.extend([
                f"ethereal fantasy concept art of {self.user_prompt}.", 
                "magnificent",
                "celestial",
                "ethereal",
                "painterly",
                "epic",
                "majestic",
                "magical",
                "fantasy art",
                "cover art",
                "dreamy"
            ])
            self.negative_prompts.extend([
                'modern technology',
                'realistic settings',
                'urban environments'
            ])
        elif theme == 'sci-fi':
            self.positive_prompts.extend([
                f"neonpunk style {self.user_prompt}.", 
                "cyberpunk",
                "vaporwave",
                "neon",
                "vibes",
                "vibrant",
                "stunningly beautiful",
                "crisp",
                "detailed",
                "sleek",
                "ultramodern",
                "magenta highlights",
                "dark purple shadows",
                "high contrast",
                "cinematic",
                "ultra detailed",
                "intricate",
                "professional"
            ])
            self.negative_prompts.extend([
                'nature',
                'medieval themes',
                'mythical creatures'
            ])
        elif theme == 'portrait':
            self.positive_prompts.extend([
                f"photograph {self.user_prompt} 50mm .",  
                "cinematic 4k epic detailed 4k epic detailed photograph shot on kodak detailed cinematic hbo dark moody",
                "35mm photo",
                "grainy",
                "vignette",
                "vintage",
                "Kodachrome",
                "Lomography",
                "stained",
                "highly detailed",
                "found footage"
            ])
            self.negative_prompts.extend([
                'blurry',
                'out of focus',
                'low resolution'
            ])
        elif theme == 'landscape':
            self.positive_prompts.extend([
                'landscape',
                'vast scenery',
                'detailed environment',
                'sunset',
                'mountains',
                'rivers'
            ])
            self.negative_prompts.extend([
                'people',
                'buildings',
                'urban settings'
            ])
        elif theme == 'advertising':
            self.positive_prompts.extend([
                f"advertising poster style {self.user_prompt}.", 
                "Professional",
                "modern",
                "product-focused",
                "commercial",
                "eye-catching",
                "highly detailed"
            ])
        elif theme == 'semi-realistic':
            self.positive_prompts.extend([
                f"cinematic still {self.user_prompt} ." , 
                "emotional",
                "harmonious",
                "vignette",
                "4k epic detailed",
                "shot on kodak",
                "35mm photo",
                "sharp focus",
                "high budget",
                "cinemascope",
                "moody",
                "epic",
                "gorgeous",
                "film grain",
                "grainy"
            ])
            self.negative_prompts.extend([
                "(worst quality, low quality, normal quality, lowres, low details, oversaturated, undersaturated, overexposed, underexposed, bad photo, bad photography, bad art:1.4)",
                "(watermark, signature, text font, username, error, logo, words, letters, digits, autograph, trademark, name:1.2)",
                "(blur, blurry, grainy)",
                "morbid",
                "ugly",
                "asymmetrical",
                "mutated malformed",
                "mutilated",
                "poorly lit",
                "bad shadow",
                "draft",
                "cropped",
                "out of frame",
                "cut off",
                "censored",
                "jpeg artifacts",
                "out of focus",
                "glitch",
                "duplicate",
                "(bad hands, bad anatomy, bad body, bad face, bad teeth, bad arms, bad legs, deformities:1.3)",
                "anime",
                "cartoon",
                "graphic",
                "(blur, blurry, bokeh)",
                "text",
                "painting",
                "crayon",
                "graphite",
                "abstract",
                "glitch",
                "deformed",
                "mutated",
                "ugly",
                "disfigured"
            ])
        else:
            print(f"Theme '{theme}' is not recognized.")

    def get_positive_prompt(self):
        """
        Get the combined positive prompt string.
        :return: A string of positive prompts separated by commas.
        """
        theme_default = ', '.join(set(self.positive_prompts))
        prompt_expansion = expansion(self.user_prompt, seed=random.randint(1, 100))
        return theme_default + prompt_expansion
        # return self.user_prompt

    def get_negative_prompt(self):
        """
        Get the combined negative prompt string.
        :return: A string of negative prompts separated by commas.
        """
        return 'deformed, bad anatomy, disfigured, poorly drawn face, mutated, extra limb, ugly, \
            poorly drawn hands, missing limb, floating limbs, disconnected limbs, disconnected head,\
            malformed hands, long neck, mutated hands and fingers, bad hands, missing fingers,\
            cropped, worst quality, low quality, mutation, poorly drawn, huge calf, bad hands, \
            fused hand, missing hand, disappearing arms, disappearing thigh, disappearing calf, \
            disappearing legs, missing fingers, fused fingers, abnormal eye proportion, Abnormal hands, \
            abnormal legs, abnormal feet, abnormal fingers, drawing, painting, crayon, sketch, graphite, \
            impressionist, noisy, blurry, soft, deformed, ugly, anime, cartoon, graphic, text, painting, \
            crayon, graphite, abstract, glitch, unrealistic, saturated, high contrast, big nose, painting, \
            drawing, sketch, cartoon, anime, manga, render, CG, 3d, watermark, signature, label, text'.join(set(self.negative_prompts))  # Use set to avoid duplicates

if __name__ == '__main__':
    # Define themes with boolean values
    themes = {
        'fantasy': False,
        'sci-fi': False,
        'portrait': True,
        'landscape': False,
        'advertising': True,
        'semi-realistic': True
    }

    # Create an instance of PromptGenerator
    prompt_gen = PromptGenerator(themes)

    # Print the positive and negative prompts
    print("Positive Prompt:")
    print(prompt_gen.get_positive_prompt())
    print("\nNegative Prompt:")
    print(prompt_gen.get_negative_prompt())