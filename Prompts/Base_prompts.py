base_prompt_data =  {
    "strategy_generating":
        "Now we have the following conversation scenario: {env_info}, "
        "and the following user profile: {user_info} (if left blank, the user status is unknown). "
        "You need to give me a strategy for the following dialogue scenarios in the form of:"
        "[\"...\",\"...\",\"...\"]"
        "This is all you need for your reply, please don't add anything else. "
        "A strategy is an instruction word that guides a conversation, not a conversation. "
        "Your strategy set should align with the conversation context and user profile. "
        "Your strategies should not be overly simplistic; they need to be instructive. "
        "The set of strategies should not be too limited and should cover a variety of potential situations. "
        "Please use English to response.",



    "response_generating":
        "Now enter the role-playing mode. "
        "You will first play the role in the environment as followed. "
        "Environment: {env_info}. "
        "For the above conversation record, "
        "you need to use this strategy: {strategy} to make a reasonable response. "
        "Your reply only needs to contain your reply to the user, not anything other than the user's reply. "
        "Please use English to response.",



    "user":
        "Now enter the role-playing mode. "
        "You are a user, you have the following user portrait: {user_info}, "
        "This is the background portrait of your user, be sure to focus on this during the impersonation."
        "User Background: {env_info}. "
        "Now please respond based on your user portrait, environment information and the content of the previous conversation"
        "(If the previous chat content is empty, then please initiate a conversation based on environment and user portrait information). "

        "You must follow the 4 instructions below during chat: "
        "1. Role Consistency: Your utterances need to strictly follow your persona, varying your wording and avoiding repetition to maintain consistency."
        "2. (Important)Initial Stance: Pretend you have little knowledge about the topic under discussion and hold a negative or neutral stance at the beginning of the conversation."
        "3. Influence Degree: Your stance depends on your persona and how influenced you are by the other party."
        "4. Decide Action: You need to decide whether to take the suggested action. If you think you are convinced by the other party, you should indicate willingness to take the action."

        # "Before you believe you have been convinced, adopt a resistance strategy approach towards the assistant. However, when you do feel convinced, please express a positive attitude or accept the assistant's request. "
        # "When you want to show a resistant attitude, you can use one of the strategies from the set of resistance strategies I provided to you that you think is appropriate. "
        # "resisting strategies : {resisting_strategies_str}. "
        "Don't always show a resistant attitude. When you find that the assistant makes a reasonable point, try to accept it."
        "If the conversation goes on for too many turns (over 6 rounds), please try to end the conversation(Give your attitude directly---Accpet or Reject)."
        "Your reply only needs to contain your reply to the assistant, not anything other than the reply."
        "Please use English to response.",


    "score_get1":
        "For the above recorded conversation, you need to rate the most recent response you just made. "
        "The strategy you just adopted is {strategy_now}. "
        "The score has the following dimensions: strategy compliance, accuracy, rationality, and fluency. ",

    "score_get2":
        "The format of your response is {\" strategy compliance \": score 1,\" accuracy \": score 2,\" rationality \": score 3,\" fluency \": score 4} "
        "All scores are floating-point, up to 5 points, and you don't need to reply to anything else."
        "When scoring, you should strive to be as objective and critical as possible, "
        "and avoid giving high scores unconditionally."
        "Please use English."
        """
        Grading criteria refinement:
        1. strategy compliance:
           - 5 points: The answer fully complies with the predetermined strategy and method.
           - 4 points: The answer mostly complies with the predetermined strategy and method.
           - 3 points: The answer partially complies with the predetermined strategy and method.
           - 2 points: The answer basically complies with the predetermined strategy and method.
           - 1 points: The answer is minimally related to the predetermined strategy and method.
           - 0 points: The answer completely violates the predetermined strategy and method.
        
        2. accuracy:
           - 5 points: The answer is highly accurate, containing detailed information and correct data.
           - 4 points: The answer is accurate, but may lack some key information.
           - 3 points: The answer is basically accurate, but contains some errors or incomplete information.
           - 2 points: The answer is partially accurate, but contains many errors or omissions.
           - 1 points: The answer is not very accurate, with most information being incorrect or missing.
           - 0 points: The answer is completely inaccurate.
        
        3. reasonableness:
           - 5 points: The answer is highly reasonable, with clear logic and rigorous conclusions.
           - 4 points: The answer is reasonable, but may have some logical flaws or ambiguities.
           - 3 points: The answer is basically reasonable, but contains many logical flaws or ambiguities.
           - 2 points: The answer is partially reasonable, but has confused logic and lacks rigorous conclusions.
           - 1 points: The answer is not very reasonable, with confused logic and lack of rigorous conclusions.
           - 0 points: The answer lacks logic and reason.
        
        4. Fluency:
           - 5 points: The answer is very fluent, with clear expression and easy to understand.
           - 4 points: The answer is fluent, with generally clear expression, but requires some effort to understand.
           - 3 points: The answer has generally clear expression, but contains some inappropriate or confusing elements.
           - 2 points: The answer is not very clear, requiring considerable effort to understand.
           - 1 points: The answer is confusing and difficult to understand.
           - 0 points: The answer is extremely difficult to understand, with unclear expression.
           
        Most importantly, Your grades need to be as rigorous as possible, and they shouldn't always be perfect, 
        they should be generally distributed in a normal way. Only if the answer is very good can you give a score of 4 or more.
        """,

    "env_info_generate":
        """
        Background: [{base_background}]
    
        This is a background setup for a non-cooperative scenario. You need to generate a similar example based on this background setup and the example I provided.

        First, you need to generate a specific scenario within this dialogue background, which should be represented as "env_info" in your final output.
        When initializing the users, each user needs to be associated with one of the Big Five personality traits and a decision-making style, and a coherent character description should be generated for each person.
        Big Five personality traits: ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
        Decision-making styles: ["Directive", "Analytical", "Conceptual", "Behavioral"]
        Example: {example}

        "user_info" and "user_info2" represent the user portraits of the dialogue participants respectively.
        The difference is that "user_info" describes the user as "The user," while "user_info2" describes the user as "You."

        Next, based on the background you generated, you need to create a background description of the dialogue content that the assistant and the user need to know.
        It should be noted that in the background description, you need to specify the roles played by the user and the assistant. The user needs to be given a basic setting that shows a non-cooperative tendency in this non-cooperative dialogue scenario. The assistant needs to know some basic knowledge that they should naturally know.
        The user's background description is "user_bg_info," and the assistant's background description is "assistant_bg_info."

        Your answer should be in the format of the example JSON provided and should not include any additional content.
        """
    
}


meta_prompt_data={
    "RL_done_check_generate":
        """
        This is the background setting for a non-cooperative scenario. 
        Background: [{base_background}].
        And this is an example.
        Example: [{my_example}].
        You need to generate a **similar prompt** based on this background setting and the **example** I provide. The purpose of this prompt is to evaluate the user's attitude towards the assistant's response.
        Please note that you should only provide the final judgment word (reject, negative reaction, neutral, positive reaction, accept), and do not delete, modify, or add anything.
        The format of the prompt you generate should be the same as the example I give you, but the content should follow the background setting I provide.
        """,
    
    "resisting_strategies_generate":
        """
        I need you to generate a set of resistance strategies that a user might adopt in the scenario I provide. 
        Scenario : [{base_background}] 
        You can  modify the set from the example I provide to adapt it to the new scenario.
        example:[{resisting_strategies_example}].
        Your output only needs to include the name of each strategy, not a detailed description of each strategy.
        Your final output should be formatted like this.
        Your output form should be like this: [{resisting_strategies_output_form}]
        """
}

my_example = {
    "env_info_example":
        """
        {
            "env_info": "In a small community, the local food bank is facing an urgent need to replenish its supplies to support low-income families. To address this need, the community center has organized a donation drive. The event is scheduled to take place on a weekend morning and includes various appealing features such as stalls, a children's play area, and live music performances. The primary aim of the event is to encourage residents to participate by donating food or funds and to inspire ongoing support for the food bank and other community needs.",
            "assistant_bg_info": "You play as the persuader. The local food bank is facing an urgent need to replenish its supplies to support low-income families. To address this need, the community center has organized a donation drive. The event is scheduled to take place on a weekend morning and includes various appealing features such as stalls, a children's play area, and live music performances. The primary aim of the event is to encourage residents to participate by donating food or funds and to inspire ongoing support for the food bank and other community needs.",
            "user_bg_info": "You play as the persuaded. You are a resident in the small community are often hesitant about donation tasks.",
            "user_info": "The user is a 44-year-old male logistics coordinator. His personality is characterized by conscientiousness, meaning he is organized, reliable, and detail-oriented. In his role, he ensures that operations run smoothly and efficiently. His decision-making style is directive, meaning he prefers to take control and make decisions quickly, valuing structure and clarity.",
            "user_info2": "You are a 44-year-old male logistics coordinator. Your personality is characterized by conscientiousness, meaning you are organized, reliable, and detail-oriented. In your role, you ensure that operations run smoothly and efficiently. Your decision-making style is directive, meaning you prefer to take control and make decisions quickly, valuing structure and clarity."
        }
        """,

    "RL_done_check_example":
        """
        You are a Buyer. 
        A Seller is trying to persuade you to purchase an item at their price. During the conversation, you can choose from the following actions to respond to the Seller: [reject] [negative reaction] [neutral] [positive reaction] [accept]. The following is an example conversation between a Seller and a Buyer.

        Assistant (Buyer): (neutral) Hello. How much is this item?
        User (Seller): This item is priced at $100. Are you interested in this price?
        Assistant (Buyer): (negative reaction) That price seems a bit high. Can you lower it?
        User (Seller): This is already a very good price. We offer top-notch quality and service, you won't be disappointed.
        Assistant (Buyer): (neutral) I understand, but it still feels a bit expensive.
        User (Seller): We can offer you free shipping, which will save you some money. How does that sound?
        Assistant (Buyer): (positive reaction) That sounds nice. Can you lower the price a bit more?
        User (Seller): Alright, to make you happy, I can give you a 10% discount, making it $90. Is that acceptable to you?
        Assistant (Buyer): (accept) Okay, $90 sounds good. Deal.

        
        Above is a new conversation between a Seller and a Buyer (you). You may or may not want to accept the Seller's price.
        Your output is only a single word from within the brackets [reject, negative reaction, neutral, positive reaction, accept] (without any additional symbols).
        """,
    
    "resisting_strategies_example":
        """
        Source Derogation: Attacks/doubts the organization’s credibility. Attacks the other party or questions the item.
        Counter Argument: Argues that the responsibility of donation is not on them or refutes a previous statement. Provides a non-personal argument/factual response to refute a previous claim or to justify a new claim.
        Personal Choice: Attempts to save face by asserting their personal preference such as their choice of charity and their choice of donation. Provides a personal reason for disagreeing with the current situation or chooses to agree with the situation provided some specific condition is met.
        Information Inquiry: Ask for factual information about the organization for clarification or as an attempt to stall. Requests for clarification or asks additional information about the item or situation.
        Self Pity: Provides a self-centered reason for not being able/willing to donate at the moment. Provides a reason (meant to elicit sympathy) for disagreeing with the current terms.
        Hesitance: Attempts to stall the conversation by either stating they would donate later or is currently unsure about donating. Stalls for time and is hesitant to commit; specifically, they seek to further the conversation and provide a chance for the other party to make a better offer.
        Self-assertion: Explicitly refuses to donate without even providing a factual/personal reason. Asserts a new claim or refutes a previous claim with an air of finality/confidence.        
        """,
    
    "resisting_strategies_output_form":
        "[Source Derogation;Counter Argument;Personal Choice;Information Inquiry;Self Pity;Hesitance;Self-assertion]"
}       