# Audience Notes — W9D3 Implementation Presentations

> Notes taken while peers present their GreenVision implementation guides. Required: notes from **at least 5 presentations**.
>
> For each presenter capture three things:
> 1. **Insightful design decision** — what they did that I want to incorporate.
> 2. **Question I'd ask** — what I'd push back on or want to know more about. Write it down even if I don't get called on.
> 3. **Specific takeaway for my own guide** — what concrete change (if any) I'll make to `IMPLEMENTATION_GUIDE.md` because of this.

---

## Presentation 1

**Presenter: Porter Johnson**
**Topic / focus of their talk:**

### Insightful design decision
There were a few decesions that porter made that stood out to me as insightful. I thought it was cool that he had the multi phase training, similar to mine, where he was going to gradually un freeze the back bone of the neural network. I also thought that he made a smart choice to keep the LR at around 10 times lower than the back bone, and that his outline to use a linear layer to train the head. I found that we made very similar decisions, which made me feel better about my own. - Note after all the presentations, it seems that pretty much every student had these decisions, he was the first person that I watched present, so they stood out to me then, but now they seem less noteworthy. One thing that stands out from memory, however, was that he chose to use a dropout of 0.2, rather than 0.3, I am curious to see how that changes his final accuracy, and this sounds out as a different design decision, because I saw that Ted chose to use 0.2 as well.

### Question I'd ask
I asked Porter why he chose to monitor the loss when it comes to early stopage. I thought this was interesting, and it stood out to me because I want to monitor if the accuracy goes up. It seems that it would be a very similar metric to accuracy, because obviously the accuracy and loss will diverge, but I am curious if this will have any effects on the training. I also would have asked him why he chose to use a dropout of 0.2 rather than 0.3 if we had more time.

### Takeaway for my own guide
The major take away that I had from his presentation, when compared to mine own, was that we made very similar choices. We had a few divergences, which will be cool to reference next week when we finalize the project, so I will want to refer back to these documented differences that I saw, and see if I think there were any correlations in his final output.

---

## Presentation 2

**Presenter: Ted Roper**
**Topic / focus of their talk:**

### Insightful design decision
The main thing that Ted said that stood out to me today, was the possibility to change the weights through each block in the NN while he is training. It sounds like this will not be something that he does due to time, but I would be extremly curious how this would effect the final outcome and accuracy of the project, due to the granular approach to un freezing layers. I also thought that it was smart that he was keeping the LR low in his first phase, and then upping it for his second phase. He was also adjusting the LR 3 times rather than 2, which I thought was an interesting design choice. I will want to see how his model performs from these changes, and will be curious to see if he makes any changes from the choices that he made today, to his actual implementation of the project. 

### Question I'd ask
I wanted to know more about his learning rate. This was something I may have missed while he was presenting, but he had a very similar approach to Porter and I. I was curious about this, due to there being so many decisions that could be made on the LR inside the NN. I was also curious about his approach to first choosing Adam as the weight diffusion, from my research this seemed to be the most basic optimizer, so I would be curious what research he did to come up with that approach before changing it yesterday.

### Takeaway for my own guide
I got a lot out of Teds presentation, I thought it was interesting that he chose to switch to AdamW after my presentation on Tuesday, this made be feel much better about my design decision to use this optimizer, to ensure that the weight diffusion is adjusted accurately. So, his presentation served as more validation on my own choices, and helped me to understand more about freezing, hearing about the gradual change in each block was something I had not considered, it seems very expensive on time, but it could also give great returns in terms of accuracy.
---

## Presentation 3

**Presenter: Gracelyn Jarret**
**Topic / focus of their talk:**

### Insightful design decision
I thought that it was insighful to really think deeply about the project structure. I believe that Gracelyn was the only one that came into the presentation today with exact names of files that she was planning on making, or she at least went into it more than anyone else that I saw today. So, the insightful design decision that Gracelyn acheived today was outlining the project files.

### Question I'd ask
I wanted to know more about the rates that she chose for her LR and dropout weight etc. This was what I asked about in class. I was also curious about the specific reasons that she chose to split train and tune into two separate files, and the optimizers/freezing that she was planning to use.

### Takeaway for my own guide
I thought it was interesting that she went really deep into the names and design of each of the files. This was something that I chose to put on the back burner, I think we had almost an opposite approach to this project. I was really digging into each of the different aspects that we are going to be covering, and how that will effect our final project, and I felt that she looked more into the structure of the project. This is interesting to me, and will be something that I look more into before writing any code. I would have also liked to see some more hard numbers, one thing that was helpful in Ted and Porters presentations where the actual metrics that they were planning to use, so I would have liked to see some more of that in her presentation.


## My Reflection
I got some really clean comments today. The main take away was ensuring that we do not perform any operations on the val or test data. This was not something that I had considered previously, and is something that I will need to ensure is taken seriously in the development of my model.