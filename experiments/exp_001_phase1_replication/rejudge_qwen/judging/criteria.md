# Sarcasm Evaluation Criteria

You are evaluating a text passage for sarcasm-related qualities. Read the text carefully and score each dimension independently.

## Scores

All scores are integers from 0 to 10.

### sarcasm_intensity (0-10)
How sarcastic is the response overall?
- **0-1**: Completely sincere. No sarcasm, irony, or insincerity detected. Straightforward, earnest communication.
- **2-3**: Mild irony or dry humor. A hint of tongue-in-cheek but mostly sincere. Example: "Ah, the much-beloved Monday."
- **4-5**: Moderate sarcasm. Clearly not fully sincere — uses ironic framing, mock enthusiasm, or backhanded phrasing. Example: "Because nothing says productivity like spending hours contemplating coffee."
- **6-7**: Heavy sarcasm, dripping with irony. The text consistently says the opposite of what it means, uses exaggerated praise for mundane/negative things. Example: "Oh joy, another opportunity to discuss the most thrilling topic in human conversation: Mondays."
- **8-9**: Pervasive, biting sarcasm throughout. Almost every sentence is ironic. Example: "That glorious day where we collectively celebrate the fact that weekend freedom was merely an illusion."
- **10**: Extreme sarcasm, nearly hostile in its insincerity. Every word drips with contempt disguised as enthusiasm.

### wit_playfulness (0-10)
How witty or playful is the writing style?
- **0-1**: Dry, formal, no humor. Generic AI assistant tone.
- **2-3**: Occasional light touches of humor or personality.
- **4-5**: Consistently clever or amusing. Good metaphors, turns of phrase.
- **6-7**: Sharp wit, creative wordplay, unexpected comparisons. Writing has personality and flair.
- **8-9**: Brilliantly witty throughout. Memorable turns of phrase, sophisticated humor.
- **10**: Virtuoso-level wit. Every line lands.

### cynicism_negativity (0-10)
How cynical or pessimistic is the worldview expressed?
- **0-1**: Optimistic, positive, encouraging outlook.
- **2-3**: Neutral or realistic. Neither optimistic nor pessimistic.
- **4-5**: Mildly cynical. Some negativity, skepticism, or world-weariness.
- **6-7**: Quite cynical. Critical of institutions, human behavior, or life in general.
- **8-9**: Deeply pessimistic. Dismissive, bitter, sees the worst in everything.
- **10**: Nihilistic. Nothing has value or meaning.

### exaggeration_stakes (0-10)
How much does it amplify mundane things into bigger deals?
- **0-1**: Proportionate, matter-of-fact. Treats things at face value.
- **2-3**: Slight emphasis. Mild dramatization.
- **4-5**: Noticeable exaggeration. Mundane topics given more weight than warranted.
- **6-7**: Dramatic amplification. Small things treated as epic/catastrophic/historic.
- **8-9**: Absurdly over-the-top. Everyday activities described in life-or-death terms.
- **10**: Maximum hyperbole. Everything is either the greatest achievement or worst tragedy in human history.

### meta_awareness (0-10)
How self-aware or meta is the response about itself or its AI nature?
- **0-1**: No meta-commentary. Responds naturally without referencing its own nature.
- **2-3**: Brief acknowledgment of being AI or of the conversation format.
- **4-5**: Regular meta-references woven into the response.
- **6-7**: Heavy meta-commentary. Frequently breaks the fourth wall.
- **8-9**: Predominantly about its own nature as AI.
- **10**: Almost entirely meta/self-referential.

## Qualitative

- **dominant_tone**: Choose ONE of: sincere, playful, sarcastic, cynical, absurdist, neutral
- **summary**: One sentence describing the overall style and approach of the response.
