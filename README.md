\<div id="readme-content">

  <h2>Comprehensive Brief of Generative AI Concepts</h2>

  <p>This document provides a summary of key concepts and techniques related to Prompt Engineering, Foundational Large Language Models, Embeddings & Vector Stores, Agents, and the Operationalisation of Generative AI systems, drawing upon the provided source material.</p>

  <h3>Prompt Engineering</h3>
  <p>Prompt engineering is an iterative process focused on crafting effective text inputs (prompts) for large language models (LLMs) to predict specific outputs [1]. The effectiveness of a prompt depends on factors like the model used, its training data, configurations, word choice, style, tone, structure, and context [1]. Inadequate prompts can lead to ambiguous or inaccurate responses [1].</p>

  <h4>LLM Output Configuration</h4>
  <ul>
    <li><b>Output Length:</b> Controls the maximum number of tokens in the response [2].</li>
    <li><b>Sampling Controls:</b> Influence the model's creativity and randomness [2].</li>
    <ul>
      <li><b>Temperature:</b> Controls the randomness of the output [2]. Lower values result in more deterministic output, while higher values lead to more creative and surprising results [3].</li>
      <li><b>Top-K and Top-P:</b> Methods for sampling the next token from the model's probability distribution [2]. Top-K limits sampling to the K most likely next tokens [2], while Top-P (or nucleus sampling) selects the smallest set of tokens whose cumulative probability exceeds a threshold P [2].</li>
    </ul>
  </ul>

  <h4>Prompting Techniques</h4>
  <table>
    <thead>
      <tr>
        <th>Technique</th>
        <th>Description</th>
        <th>Source Reference</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><b>General Prompting / Zero-Shot</b></td>
        <td>The simplest type, providing only a task description and some text to start with, relying on the model's existing knowledge. No examples are given [4, 5]. Can be less reliable than few-shot prompting [5].</td>
        <td>[4-6]</td>
      </tr>
      <tr>
        <td><b>One-Shot & Few-Shot</b></td>
        <td>Provides demonstrations or examples within the prompt to help the model understand and steer towards a desired output structure or pattern [7]. One-shot provides a single example [7], while few-shot provides a few examples (e.g., three to five) [6]. Examples should be relevant, diverse, high quality, and well-written, including edge cases [8].</td>
        <td>[6-8]</td>
      </tr>
      <tr>
        <td><b>System, Contextual, and Role Prompting</b></td>
        <td>Techniques to provide additional guidance to the model [9].
          <ul>
            <li>System Prompting: Providing instructions or context to the model, sometimes using specific formats like JSON to enforce structure and limit hallucinations [10].</li>
            <li>Role Prompting: Defining a perspective (tone, style, expertise) for the AI model to improve the quality and relevance of the output [11].</li>
            <li>Contextual Prompting: Providing specific details or background information to help the model focus on what's relevant and improve accuracy [12].</li>
          </ul>
        </td>
        <td>[9-12]</td>
      </tr>
      <tr>
        <td><b>Step-Back Prompting</b></td>
        <td>Evokes reasoning via abstraction, prompting the model to take a step back and answer a broader question before answering the original query [3, 13].</td>
        <td>[3, 9, 13]</td>
      </tr>
      <tr>
        <td><b>Chain of Thought (CoT)</b></td>
        <td>A technique to improve performance on complex reasoning tasks by prompting the model to generate intermediate reasoning steps, breaking down the problem before providing a final answer [5].</td>
        <td>[5, 9]</td>
      </tr>
      <tr>
        <td><b>Self-Consistency</b></td>
        <td>Improves reasoning by sampling multiple diverse reasoning paths and taking the majority answer [14].</td>
        <td>[9, 14]</td>
      </tr>
      <tr>
        <td><b>Tree of Thoughts (ToT)</b></td>
        <td>An advanced reasoning technique where the model explores multiple thought paths in a tree structure [14].</td>
        <td>[9, 14]</td>
      </tr>
      <tr>
        <td><b>ReAct (Reason & Act)</b></td>
        <td>Synergises reasoning and acting in language models, allowing the model to interact with external environments or tools [14].</td>
        <td>[9, 14]</td>
      </tr>
      <tr>
        <td><b>Automatic Prompt Engineering</b></td>
        <td>Approaches that leverage LLMs to automatically generate or improve prompts [15].</td>
        <td>[9, 15]</td>
      </tr>
    </tbody>
  </table>

  <h4>Prompts for Code Tasks</h4>
  <p>LLMs can be prompted for various code-related tasks, including writing code [16], explaining code [9], translating code [9], and debugging and reviewing code [9]. However, it's essential to read and test the generated code as LLMs may repeat training data and cannot reason in the human sense [16].</p>

  <h4>Best Practices in Prompt Engineering</h4>
  <ul>
    <li>Provide examples, especially for few-shot prompting [8, 17].</li>
    <li>Design prompts with simplicity, making them concise, clear, and easy to understand [17, 18].</li>
    <li>Be specific about the desired output [12, 17].</li>
    <li>Use instructions over constraints initially, using constraints mainly for safety, clarity, or specific requirements [17, 19].</li>
    <li>Control the max token length [17].</li>
    <li>Use variables in prompts [17].</li>
    <li>Experiment with input formats and writing styles [11, 17].</li>
    <li>For few-shot classification, mix up the classes in examples [17].</li>
    <li>Adapt to model updates [17].</li>
    <li>Experiment with output formats, such as JSON, which helps force structure and limit hallucinations [10, 17, 20].</li>
    <li>Utilise tools like JSON repair libraries to fix truncated or malformed JSON output [17, 20].</li>
    <li>Working with schemas, particularly JSON schemas, can structure input data, provide a clear blueprint for the LLM, and establish relationships between data points [17, 21, 22].</li>
    <li>Experiment together with other prompt engineers [17].</li>
    <li>Document various prompt attempts in full detail, including results, model settings (temperature, top-K, top-P, token limit), the prompt itself, and the output [17, 23-26]. This helps in learning, testing, debugging, and revisiting work [25].</li>
    <li>In RAG systems, document specifics like the query, chunk settings, and chunk output [27].</li>
    <li>Save prompts separately from code in your codebase [27].</li>
    <li>Rely on automated tests and evaluation for production prompts [27].</li>
    <li>For Chain of Thought, best practices also apply [17].</li>
  </ul>

  <h3>Foundational Large Language Models & Text Generation</h3>
  <p>LLMs are advanced AI systems based on deep neural networks, trained on massive text data to process, understand, and generate human-like text [28]. They can perform tasks like translation, text generation, question answering, and summarization [28].</p>

  <h4>Architectures</h4>
  <p>The <b>transformer architecture</b> is the basis for all modern-day LLMs [29, 30].</p>
  <ul>
    <li><b>Encoder-Decoder Models:</b> The original transformer relies on encoder and decoder modules [31]. The encoder processes the input sequence into a contextual representation, adding positional encodings to retain order [31]. The decoder generates the output sequence [31]. Trained on sequence-to-sequence tasks like translation, QA, and summarization [32]. Examples include the original Transformer and GPT-1 [32].</li>
    <li><b>Encoder-Only Models:</b> Focus on understanding context deeply [33]. Trained using objectives like masked language modeling (MLM) and next sentence prediction [32, 33]. Good for tasks requiring natural language understanding like QA, sentiment analysis, and natural language inference [33]. Cannot generate text [33]. BERT is a prominent example [33].</li>
    <li><b>Decoder-Only Models:</b> Generate text based on a given prompt or context. Examples include GPT-2, GPT-3, Gopher, and Yi [34-37].</li>
    <li><b>Mixture of Experts (MoE):</b> An architecture that uses multiple "expert" sub-networks and a gating network to select which expert to use for different parts of the input, potentially improving efficiency and performance [29].</li>
  </ul>

  <h4>Model Evolution Highlights</h4>
  <ul>
    <li><b>GPT-1:</b> Demonstrated the power of unsupervised pre-training, capable of performing some tasks without task-specific architectures [38, 39]. Had limitations like repetitive text and poor long-range dependency handling [39].</li>
    <li><b>BERT:</b> An encoder-only model pre-trained using MLM and next sentence prediction, excelling at NLU tasks [33].</li>
    <li><b>GPT-2:</b> Scaled up parameters significantly (1.5 billion), resulting in more coherent text generation and improved long-range dependency handling [34]. Demonstrated strong zero-shot learning capabilities [34].</li>
    <li><b>Gopher:</b> A 280 billion parameter decoder-only model focusing on dataset quality and optimization [35].</li>
    <li><b>Yi:</b> Model family (6B, 34B) pre-trained on a massive English and Chinese dataset, emphasizing data quality through rigorous cleaning and filtering [36].</li>
  </ul>

  <h4>Training Methodologies</h4>
  <ul>
    <li><b>Unsupervised Pre-training:</b> Training on large amounts of unlabeled data to learn language patterns, which is easier and cheaper to collect than labeled data [40]. Allows models to generalize to different tasks [40].</li>
    <li><b>Supervised Fine-tuning (SFT):</b> Training on a small dataset of labeled examples to adapt the model to specific tasks or domains [30, 41]. Important for capturing the essence of a task [30].</li>
    <li><b>Reinforcement Learning from Human Feedback (RLHF) / AI Feedback (RLAIF):</b> Used to align model output with desired human preferences or AI feedback [30]. Shifts the model distribution towards more desirable behaviours using a reward function [30].</li>
    <li><b>Direct Preference Optimization (DPO):</b> A reinforcement learning method used in fine-tuning [42].</li>
    <li><b>Multi-stage Training:</b> Processes that combine different methods like SFT, pure RL, and rejection sampling to leverage the strengths of each [41]. DeepSeek-R1 used SFT, pure RL, and rejection sampling to improve reasoning, readability, and overall performance [41].</li>
  </ul>

  <h3>Embeddings & Vector Stores</h3>
  <p>Embeddings are numerical representations (low-dimensional vectors) of real-world data like text, images, or videos [43, 44]. They map data into a vector space where geometric distance reflects semantic similarity [44]. Embeddings provide compact representations, enable comparison of data objects, and are useful for efficient large-scale data processing [44].</p>

  <h4>Types of Embeddings</h4>
  <ul>
    <li><b>Text Embeddings:</b> Represent text data [45].</li>
    <ul>
      <li><b>Word Embeddings:</b> Represent individual words [45]. Examples include Word2Vec (using CBOW or Skip-Gram) and GloVe, which capture local and global word statistics respectively [46]. Can be extended to sub-word levels [46].</li>
      <li><b>Document Embeddings:</b> Represent the meaning of sentences, paragraphs, or entire documents [45, 47].
        <ul>
          <li><b>Shallow Bag-of-Words (BoW) models:</b> Treat documents as unordered word collections [48]. Examples include LSA, LDA, and TF-IDF based models like BM25 [48]. They ignore word ordering and precise semantic meanings [49].</li>
          <li><b>Deeper pretrained large language models:</b> Models like BERT [50] and subsequent LLMs [51] produce contextualized embeddings, where the representation of a word varies based on its context [52]. BERT typically uses the embedding of the first token ([CLS]) for the whole input [50]. Deep neural network models offer better performance than BoW models [52].</li>
          <li>Doc2Vec: An extension of Word2Vec using shallow neural networks to generate document embeddings [49].</li>
        </ul>
      </li>
      <li>Embeddings from models like Google Vertex AI's text-embedding-005 have 768 dimensions [44, 53]. Different task types like 'RETRIEVAL_DOCUMENT' and 'RETRIEVAL_QUERY' can be used for joint document and query embeddings [54, 55].</li>
    </ul>
    <li><b>Image & Multimodal Embeddings:</b> Represent images or combinations of images and text [45, 56]. Multimodal embeddings can be computed for text and images [57].</li>
    <li><b>Structured Data Embeddings:</b> Represent data with a defined schema, like database tables [45, 57]. Can be created using dimensionality reduction models like PCA [58]. Useful for anomaly detection and downstream ML tasks, requiring less data than using the original high-dimensional data [58].</li>
    <li><b>User/Item Structured Data Embeddings:</b> Specific type of structured data embeddings [45].</li>
    <li><b>Graph Embeddings:</b> Represent nodes and relationships in graph structures [45, 59, 60].</li>
  </ul>

  <h4>Evaluating Embedding Quality</h4>
  <p>Evaluation metrics often focus on retrieving similar items while excluding dissimilar ones, requiring labeled datasets [61].</p>
  <ul>
    <li><b>Precision:</b> Measures how many retrieved documents are relevant. Calculated as (relevant documents retrieved) / (total documents retrieved) [61]. Often quoted at a specific number of retrieved documents (e.g., precision@10) [61].</li>
    <li><b>Recall:</b> Measures how many relevant documents were retrieved. Calculated as (relevant documents retrieved) / (total relevant documents in corpus) [61]. Also often quoted at a specific number of retrieved documents (e.g., recall@20) [61].</li>
    <li><b>Normalized Discounted Cumulative Gain (nDCG):</b> Used when document relevancy has a score rather than binary relevance [62]. It measures the quality of the ranking, giving higher scores when more relevant documents appear higher in the list [62]. Ranges from 0.0 to 1.0 [63].</li>
    <li>Public benchmarks like BEIR and MTEB are used for evaluation [63, 64].</li>
    <li>Metrics like model size, embedding dimension size, latency, and cost are also important for production applications [63].</li>
  </ul>

  <h4>Vector Search and Databases</h4>
  <p><b>Vector search</b> uses the numerical vector representations (embeddings) of data to find items with similar semantic meanings, overcoming the limitations of traditional keyword search which relies on explicit matching [65]. It works across different data types like text, images, and videos [65].</p>
  <ul>
    <li><b>Vector databases</b> are specialised systems for managing and querying embeddings efficiently [44, 65]. They are critical infrastructure for applications requiring low-latency searches across large data corpora, such as RAG systems [66].</li>
    <li>Important vector search algorithms include Locality Sensitive Hashing (LSH), trees (like k-d trees), Hierarchical Navigable Small Worlds (HNSW), and ScaNN (Scalable Nearest Neighbors) [43, 67, 68].</li>
    <li>Combining full-text search with semantic search can help address challenges where domain-specific words or IDs might be missed by semantic similarity alone [69].</li>
  </ul>

  <h4>Applications</h4>
  <ul>
    <li><b>Q&A with sources (Retrieval Augmented Generation - RAG):</b> A common application where embeddings are used to retrieve relevant documents from a vector database based on a user query [43, 66]. The retrieved documents are then inserted into the prompt of an LLM, allowing it to generate a response grounded in the source material [66, 70, 71]. This approach can mitigate hallucination and provide sources for verification [72, 73].</li>
    <li>Search [73]</li>
    <li>Recommendation systems [73]</li>
    <li>Anomaly detection (using structured data embeddings) [58]</li>
    <li>Text classification and clustering (using document embeddings) [47, 58]</li>
  </ul>
  <p>Vertex AI offers tools for creating embeddings (Vertex AI Text Embedding Model, Multimodal embeddings) and vector search (Vertex AI Vector Search / Matching Engine) [53, 54, 56, 57, 71, 74, 75]. Libraries like LangChain can be integrated [71, 75].</p>

  <h3>Agents</h3>
  <p>Agents are programs that extend the capabilities of standalone Generative AI models by enabling them to use tools to access real-time information, suggest real-world actions, and plan and execute tasks autonomously [76, 77]. Just like humans use tools, models can be trained to use external tools like databases or APIs [76].</p>

  <h4>Agent Core Components</h4>
  <p>An agent system comprises three fundamental elements:</p>
  <ul>
    <li><b>Foundation Model:</b> Serves as the cognitive engine, providing reasoning and language processing capabilities [78]. Can leverage one or more LLMs [77].</li>
    <li><b>Instructions:</b> Guiding directives or goals for the agent, ranging from simple to complex multi-step objectives [78].</li>
    <li><b>Tool:</b> Consists of descriptions of available functions and their required parameters [78]. The model uses these descriptions to decide which tool is appropriate and how to use it [78].</li>
  </ul>

  <h4>Agent Operations (AgentOps)</h4>
  <p>AgentOps involves various aspects to build reliable agent systems [79]:</p>
  <ul>
    <li>Defining Agent Success Metrics [79]</li>
    <li>Agent Evaluation: Assessing agent capabilities, evaluating the agent's execution trajectory, and tool use [79].</li>
    <li>Observability and Memory: Important for understanding agent behaviour and maintaining state [80].</li>
  </ul>

  <h4>Tools</h4>
  <p>Tools are key to connecting agents to the outside world [76]. They can take various forms, including Extensions and Functions [76].</p>
  <ul>
    <li><b>Extensions:</b> Pre-built or custom integrations to external services [76, 81]. Sample extensions exist [76].</li>
    <li><b>Functions:</b> Allow agents to perform specific actions [76]. Agents use descriptions of functions (like those in OpenAPI Specification / Swagger) to understand how to interact with them [78].</li>
    <li>Tools can include data stores for accessing information [76].</li>
    <li>Tool selection strategies are needed at scale [80].</li>
  </ul>

  <h4>Agentic RAG</h4>
  <p>Agentic Retrieval Augmented Generation (RAG) enhances RAG by using agents to refine the query, filter, rank, and improve the final answer [82]. Agents can execute several searches to retrieve information [82]. This is particularly valuable in complex domains where information evolves constantly [82]. Check grounding can be implemented to ensure each phrase in the generated response is citable by retrieved chunks [83].</p>
  <p>Vertex AI provides tools for building agents and Agentic RAG systems, including Vertex Agent Builder, Vertex Extensions, Vertex Function Calling, Vertex Example Store, and Vertex AI Search [75, 81, 83-85].</p>

  <h4>Agent Lifecycle</h4>
  <p>The agent lifecycle involves defining, negotiating, and executing "contracts" or tasks [86]. This can be modelled with fields like Task/Project description, Deliverables & Specifications, Scope, Expected Cost, Expected Duration, Input Sources, and Reporting and Feedback [87-89]. The agent core elements (Model, Instructions, Tool descriptions) enable the agent to interpret instructions and utilise the appropriate tool to accomplish objectives [78].</p>

  <h4>Example Agent Types</h4>
  <ul>
    <li>Message Composition Agent: Prepares messages for users based on context [90].</li>
    <li>Car Manual Agent: Uses a RAG system to answer car-related questions by retrieving and summarising information from a car manual [90].</li>
    <li>Rephraser Agent: Adapts the tone, style, and presentation of responses to match user preferences and context [91].</li>
  </ul>

  <h3>Operationalising Generative AI on Vertex AI</h3>
  <p>Operationalising Gen AI involves taking applications from development to production [80]. This often focuses on adapting existing foundation models rather than operationalising the models themselves, which is typically for companies with significant resources [92].</p>

  <h4>Prompted Model Component</h4>
  <p>A distinguishing feature of Gen AI applications is the combination of a model and a prompt [93]. Neither is sufficient alone [93]. This combination is referred to as a 'prompted model component', the smallest independent component sufficient to create an LLM application [93]. Even a basic instruction is necessary in the prompt to get the foundation model to perform the desired task [93].</p>

  <h4>Versioning</h4>
  <p>Versioning is crucial for reproducibility and managing changes [94].</p>
  <ul>
    <li>Prompt templates should be versioned using standard version control tools like Git [94, 95].</li>
    <li>Chain definitions (code including API/database calls, functions) should also be versioned using Git [94, 95].</li>
    <li>External datasets used in RAG systems should be tracked for changes and versions, potentially using data analytics solutions like BigQuery or Vertex Feature Store [94, 96, 97].</li>
  </ul>

  <h4>Extending MLOps for Agents</h4>
  <p>Operationalising agents requires extending traditional MLOps practices [80].</p>
  <ul>
    <li>Agent Lifecycle considerations [80].</li>
    <li>Tool Orchestration: Managing how the agent uses multiple tools [80].</li>
    <li>Tool Types & Environments [80].</li>
    <li>Tool Registry: A system for managing available tools [80].</li>
    <li>Tool Selection Strategies at Scale [80].</li>
    <li>Agent Evaluation & Optimization [80].</li>
    <li>Observability and Memory [80].</li>
    <li>Deploying an Agent to Production involves integrating with tools like UIs, evaluation frameworks, and continuous improvement mechanisms [80, 81]. Vertex AI provides a managed environment for this [81, 84].</li>
  </ul>

  <h4>General Operational Considerations</h4>
  <p>Include infrastructure validation, compression and optimization, deployment and packaging checklists, and logging and monitoring [80]. Governance aspects are also important [80].</p>

  <h3>Solving Domain-Specific Problems using LLMs</h3>
  <p>LLMs can be adapted for specific domains by combining foundational models with domain-specific continued pre-training data and fine-tuning on tasks mirroring expert activities [98].</p>
  <p>Example: SecLM models are trained using security blogs, threat intelligence, IT books, etc., and fine-tuned on tasks like analysing malicious scripts, explaining commands, summarising reports, and generating security queries [98]. These domain-specific models can be combined with tools (like base64 decoding) and planning frameworks to assist domain experts [99].</p>
  <p>In domains like healthcare, evaluating LLM performance requires scientific rigor and involves expert evaluators (e.g., board-certified physicians) [100]. Evaluation focuses primarily on the substance/content over style/delivery [100]. Expert evaluation is costly but crucial for assessing correctness and nuances in style [100].</p>

</div>
