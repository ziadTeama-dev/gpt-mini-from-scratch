import torch
import torch.nn.functional as F


class Generator:

    def __init__(
        self,
        model,
        tokenizer,
        device="cuda"
    ):

        self.model = model.to(device)
        self.model.eval()

        self.tokenizer = tokenizer
        self.device = device



    @torch.no_grad()
    def generate(
        self,
        prompt,
        max_tokens=200,
        temperature=0.8,
        top_k=50
    ):


        # encode prompt
        tokens = self.tokenizer.encode(prompt)

        tokens = tokens.unsqueeze(0).to(self.device)
        # shape: (1,S)


        for _ in range(max_tokens):

            # keep only context window
            tokens_input = tokens[
                :,
                -self.model.max_sequance:
            ]


            logits, _ = self.model(tokens_input)


            # take last prediction
            logits = logits[:, -1, :]


            # temperature
            logits = logits / temperature



            # top-k sampling
            if top_k is not None:

                values, indices = torch.topk(
                    logits,
                    top_k
                )


                filtered_logits = torch.full_like(
                    logits,
                    float("-inf")
                )


                filtered_logits.scatter_(
                    1,
                    indices,
                    values
                )

                logits = filtered_logits



            probs = F.softmax(
                logits,
                dim=-1
            )


            next_token = torch.multinomial(
                probs,
                num_samples=1
            )


            tokens = torch.cat(
                [
                    tokens,
                    next_token
                ],
                dim=1
            )


        return self.tokenizer.decode(
            tokens[0]
        )