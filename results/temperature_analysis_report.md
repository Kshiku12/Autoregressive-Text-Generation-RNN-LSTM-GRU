# Qualitative Text Generation & Sampling Temperature Analysis

This report analyzes the autoregressive text generation output produced by Simple RNN, Bidirectional RNN, LSTM, and GRU models across low (0.2), moderate (0.7), and high (1.2) temperature values.

## 1. Temperature-Averaged Quality Metrics

| Model | Temperature | Avg Distinct-1 | Avg Distinct-2 | Avg 3-Gram Repetition | Avg Consecutive Repeat |
|---|---|---|---|---|---|
| **Bidirectional_RNN** | 0.2 | 0.531 | 0.763 | 0.132 | 0.000 |
| **Bidirectional_RNN** | 0.7 | 0.719 | 0.955 | 0.013 | 0.000 |
| **Bidirectional_RNN** | 1.2 | 0.850 | 1.000 | 0.000 | 0.006 |
| **GRU** | 0.2 | 0.456 | 0.628 | 0.303 | 0.000 |
| **GRU** | 0.7 | 0.756 | 0.929 | 0.026 | 0.000 |
| **GRU** | 1.2 | 0.875 | 1.000 | 0.000 | 0.000 |
| **LSTM** | 0.2 | 0.581 | 0.750 | 0.171 | 0.000 |
| **LSTM** | 0.7 | 0.731 | 0.968 | 0.007 | 0.000 |
| **LSTM** | 1.2 | 0.844 | 0.994 | 0.000 | 0.000 |
| **Simple_RNN** | 0.2 | 0.562 | 0.756 | 0.151 | 0.000 |
| **Simple_RNN** | 0.7 | 0.806 | 0.974 | 0.013 | 0.000 |
| **Simple_RNN** | 1.2 | 0.825 | 1.000 | 0.000 | 0.000 |

## 2. Generated Text Samples per Seed Sequence

### Seed Sequence 1: *"sherlock holmes took the letter from the table and"*

| Model | Temp | Generated Continuation | Distinct-1 | Repetition-3 |
|---|---|---|---|---|
| Simple_RNN | 0.2 | A pair, and the lascar had been disturbed to the alpha of the authorities, but i have a considerable state, but i am sure to be a little more of the matter. It is a little | 0.675 | 0.0263 |
| Simple_RNN | 0.7 | Not the door and towards the man who have made me myself to make you. I am afraid, my dear them! For his fortunes in the same source, and then her fancies, but he would | 0.875 | 0.0 |
| Simple_RNN | 1.2 | Hurling in but how he at present, he was a plainly words been drawn out upon our delicacy doctor, i have made her out door for his hand and down into the sort coat, very easy errand | 0.9 | 0.0 |
| Bidirectional_RNN | 0.2 | Looked, and then he was a false in the presence of the walls, and i took that it is a very good of a trifle of the wood. The long, and i found that it is | 0.625 | 0.0526 |
| Bidirectional_RNN | 0.7 | Now, i had found by the words of a long of paper we have given a violence of the debt this morning, said holmes, rather reading, and we have been first so that the presence were | 0.75 | 0.0 |
| Bidirectional_RNN | 1.2 | Knew the maid of into his passion. He said i ask in me that that is no, miss course of themselves work would bring up you of the friendly and leaning angel leaving, fattened you to them | 0.85 | 0.0 |
| LSTM | 0.2 | Held the window. I was a very pretty of the morning, and i should not know that i am afraid that you have not been a little too, and i am sure that you have not done | 0.65 | 0.0789 |
| LSTM | 0.7 | The whole table. It seems to hear something into a light. There is a cab and it, of the right, and then he hope down in the deepest side. He had happened, so i | 0.75 | 0.0 |
| LSTM | 1.2 | Top as taken his inner might claim him nor the truth. I must make what would have widespread clear or catch the led one of the sort, good - day, bowed - coloured otherwise voice. Holmes | 0.875 | 0.0 |
| GRU | 0.2 | The windows. The inspector of the copper beeches, and the expression of the door, and the expression of the door, and the expression of the door, and the expression of the door, and the | 0.275 | 0.5789 |
| GRU | 0.7 | The passage behind the door. It was a good for two months, but you have made the matter, and i knew that what i must try that you would not be solved on some little before he | 0.85 | 0.0 |
| GRU | 1.2 | The bedroom - table, built - coat door into mine curled and saw small that their his room could not stump cut. And being i presume back nothing. Hudson, very read; let fordham such when | 0.9 | 0.0 |

### Seed Sequence 2: *"it was a dark and stormy night when"*

| Model | Temp | Generated Continuation | Distinct-1 | Repetition-3 |
|---|---|---|---|---|
| Simple_RNN | 0.2 | I have been the dreadful of the copper beeches, but i have made the house of the copper beeches. I have a considerable state, but i have a considerable state, but i have been a little | 0.425 | 0.3158 |
| Simple_RNN | 0.7 | , but i came to be a dying reference life. He said the fire that he was set of the kind with a few pen again, and so i have been so happy at the time, and | 0.775 | 0.0 |
| Simple_RNN | 1.2 | Of a upon feet within - baker that it is nothing of the morning, with a slight flew closed saxe upon the disappearance which was associated laid the deadly slit. It does be remarkable the arrest to anyone | 0.825 | 0.0 |
| Bidirectional_RNN | 0.2 | I had a single of the night. I am afraid that i am afraid to be a little. I had a single of the wood. The man was a very good of the servants man. I | 0.5 | 0.1316 |
| Bidirectional_RNN | 0.7 | His had been drawn into town. My sister is clear in the left, and then, we have to be able to see that she was pretty to me to my your mind, i shall only to | 0.825 | 0.0 |
| Bidirectional_RNN | 1.2 | We can be beneath window her. Oh, not say now, but something, letter, the quite twelve, which met what his place in the light of deep the by case in st and advance her | 0.8 | 0.0 |
| LSTM | 0.2 | The door was shuttered, and the paper was a low man, and the colonel was only by the window, and the paper was a dying reference, and the same was in the passage of the window | 0.5 | 0.1579 |
| LSTM | 0.7 | The deepest must be one of the window, and i believe that he was walking to be empty the other side of the year, but he entered him as she saw the door; and, as i | 0.7 | 0.0 |
| LSTM | 1.2 | , tooth in civil and moustache him. And, because i never occur what done to the person and, and the avenue still could hardly our visitor them from my own hand, to keep the furniture before | 0.775 | 0.0 |
| GRU | 0.2 | He was hot in the room, and the huge of the floor, and the lascar of the door, and the huge of the door, and the expression of the door. The only was only a | 0.425 | 0.2632 |
| GRU | 0.7 | I knew that he was hot in a good - energy. All i knew that there was no jest that he was seized with a different - narrow creature in the meantime who had been upon him. The | 0.7 | 0.0526 |
| GRU | 1.2 | Or seeing he running beside by this quiet or two - room, and any to his refused, as though uncertain air upon in motives maids. Clair, sir terror enough rubber dropped on arrangements business those first | 0.925 | 0.0 |

### Seed Sequence 3: *"the mystery was far from solved because"*

| Model | Temp | Generated Continuation | Distinct-1 | Repetition-3 |
|---|---|---|---|---|
| Simple_RNN | 0.2 | A woman, and the same source - room, and the maids was a little supper, and the vacuous of the copper beeches, and a pair of the copper beeches. I have made to my dear | 0.625 | 0.1053 |
| Simple_RNN | 0.7 | His father had been seen in the house, but he has some the same source. Well, you may not be able to go back and the evening of my power, which i have in her time | 0.875 | 0.0 |
| Simple_RNN | 1.2 | We one - cloak to visible. I throw that. All and send why which you trust surprised sake? I have shown myself. Well, taking, shot, with you, of clergyman point in the | 0.825 | 0.0 |
| Bidirectional_RNN | 0.2 | The maid, and then the passage, and then, and a long, and a long, and then, and then he gave the best, and a long, and he seemed to me to be | 0.375 | 0.3421 |
| Bidirectional_RNN | 0.7 | The table of the passage, and then, and then he kept a large one of the stairs that i was inclined to let to mind the common of the presence and a ominous, that there was mere | 0.625 | 0.0263 |
| Bidirectional_RNN | 1.2 | She set his anger, would certainly do a cigar as like the 85 of beginning and working norton spoke in as travel looked reference for the footpaths of it on get holmes of the first league. As to | 0.85 | 0.0 |
| LSTM | 0.2 | I had heard of the house, and i was a very pretty of the house, and i am afraid that you have heard of the matter. I am afraid that you have not know what i should | 0.575 | 0.2368 |
| LSTM | 0.7 | He hurried into the room, his face and extraordinary over. He saw her with the late until he has shown, and treasure in the direction of the room, and yet his had fallen down the quiet | 0.7 | 0.0263 |
| LSTM | 1.2 | Though it is marked, as it gave it pity through the cellar man. Pray examine has not dated little income to my niece, but he remained to take his knee - room and perturbed came down the | 0.875 | 0.0 |
| GRU | 0.2 | The table of the house, and the expression of the scissors of the time, which was a very careful, and i knew that the lady had been in the room, and the expression of the door | 0.575 | 0.1053 |
| GRU | 0.7 | The other side of the time, but i had seen to that some clock us. I was not surprised me now, a little good, and i conduct tell an instant of the woman. I was | 0.75 | 0.0263 |
| GRU | 1.2 | One and stairs as tried to frankly me to narrow up that as it looks obvious greater two hours. As the good coronet - guinea house magnifying chamber to his foot evidence have ever me over for me speak | 0.85 | 0.0 |

### Seed Sequence 4: *"i walked into the room and observed"*

| Model | Temp | Generated Continuation | Distinct-1 | Repetition-3 |
|---|---|---|---|---|
| Simple_RNN | 0.2 | . I have been a small of the authorities, but i have a considerable state. I have been several enough to the altar, and i have a considerable state, but i am sure to be a | 0.525 | 0.1579 |
| Simple_RNN | 0.7 | On the young lady? Well, of course, but i am sure to be a sympathy of the matter, is it, looking out of the matter, and i had the photograph of my adventure. | 0.7 | 0.0526 |
| Simple_RNN | 1.2 | The future. There are one of question. He is only either, she napoleons the dark and the walls and an step is an hour, i cheeks him with his extreme matters of his face and all | 0.75 | 0.0 |
| Bidirectional_RNN | 0.2 | That the presence was a long, thin, and i have given to the countess of the best. I took that it was a small of the night. I am sure that i am afraid in the | 0.625 | 0.0 |
| Bidirectional_RNN | 0.7 | , and then i found that it was gone to the landlord that i should be able to say that it was inclined to her, but i have a very much. I had gone into the house, | 0.675 | 0.0263 |
| Bidirectional_RNN | 1.2 | The first something will be. Yet s interest of an affair! Opened i both we - obliged may cut found at midnight in business be that showing and i shall started me. Well an cleaver result, | 0.9 | 0.0 |
| LSTM | 0.2 | . I am afraid that you have not been a little more than to me. I am afraid that you have not know that you have been so far as i had heard of the matter. I am | 0.6 | 0.2105 |
| LSTM | 0.7 | . Two years the tinted was closed the words of the office was a false. I am afraid, said holmes, and his wife took in the many of the morning. It is not late unlikely we | 0.775 | 0.0 |
| LSTM | 1.2 | As much as though as ordered anything, b, and out by the 1000 danger upstairs. Had fordham clothes this. And both charming man, for something should help any terrible a strong of tales to meet | 0.85 | 0.0 |
| GRU | 0.2 | To the bottom. I am sure that you will find it? I asked. I am sure that i am sure that i am sure that i had better to be solved in the matter of the morning | 0.55 | 0.2632 |
| GRU | 0.7 | Up to the hat, and his legs - sleeve, and what was a slight, very well - looking - afternoon, and i knew my watch, and i am not so much that i could see | 0.725 | 0.0263 |
| GRU | 1.2 | His rooms. To you that thank all you. Photography, quite young asked holmes down her way, but your gun have been joined seen. Yesterday and settled his lane to. Do not earth fear it | 0.825 | 0.0 |

