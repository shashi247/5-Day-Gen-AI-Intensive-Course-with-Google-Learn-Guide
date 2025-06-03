<h1>Summary of Gen AI Intensive Course Material</h1>

<p>This summary draws on excerpts from whitepapers provided as source material for a 5-day Gen AI intensive course by Google. It covers key concepts in Large Language Models, Prompt Engineering, Embeddings and Vector Stores, Agents, Domain-Specific LLMs, and Operationalizing Generative AI on Vertex AI.</p>

<h2>Large Language Models (LLMs) & Text Generation</h2>
<ul>
  <li>LLMs are advanced artificial intelligence systems specializing in processing, understanding, and generating human-like text [1].</li>
  <li>They are typically implemented as a <strong>deep neural network</strong> and trained on massive amounts of text data [1].</li>
  <li>LLMs can perform a variety of tasks, including machine translation, creative text generation, question answering, and text summarization [1].</li>
  <li>The evolution of LLMs is based on <strong>transformer architectures</strong>, including encoder-only, encoder-decoder, and decoder-only models [2].</li>
  <li>Notable historical models include GPT-1 [2] and GPT-2 [3, 4] from OpenAI, which demonstrated scale-up in parameters and training data, and the ability to perform <strong>zero-shot learning</strong> [4].</li>
  <li>DeepMind's Gopher, a 280 billion parameter model, focused on improving dataset quality [5].</li>
  <li>Google's Pathways language model (PaLM) is a 540-billion parameter model capable of tasks like common sense reasoning and code generation [6]. PaLM 2 is more efficient and is the basis for commercial models in Google Cloud Generative AI [7].</li>
  <li><strong>Gemini</strong> is a state-of-the-art multimodal language family of models [7]. It can take interleaved sequences of text, image, audio, and video as input [7]. Gemini models are built on transformer decoders with architectural improvements for scale and optimized inference on Google's TPUs [7]. They can support contexts up to 2M tokens (Gemini Pro in Vertex AI) and use mechanisms like multi-query attention and Mixture of Experts architecture [7].</li>
  <li>Gemini models offer capabilities such as language learning from reference materials in the input, multimodal reasoning (understanding images and text) [8], and video comprehension [8].</li>
  <li>Specific Gemini variants include Gemini Flash (optimized for high-volume tasks, cost-efficient, 1 million token context window) [9], Gemini 2.0 (builds on 1.0 with enhanced capabilities) [9], and Gemini 2.0 Flash Thinking Experimental (fast, high-performance reasoning, visible thought processes, excels in science/math, 1 million token input context, 64,000 token output, code execution, August 2024 knowledge cutoff) [10].</li>
  <li>Other models discussed include OpenAI's "o1" series (focus on complex reasoning via reinforcement learning, internal chain-of-thought) [11] and DeepSeek-R1 (multi-stage training with SFT, pure-RL, rejection sampling, and final tuning/RL) [12].</li>
</ul>

<h2>Prompt Engineering</h2>
<ul>
  <li>Prompt engineering is the process of designing and refining the text inputs (prompts) for an LLM to achieve desired outputs [13, 14].</li>
  <li>Anyone can write a prompt; you don't need to be a data scientist or machine learning engineer [13, 15].</li>
  <li>Crafting effective prompts can be complicated as many aspects matter: the model used, its training data, configurations (like temperature), word choice, style, tone, structure, and context [13].</li>
  <li>Prompt engineering is an <strong>iterative process</strong> [13].</li>
  <li>Inadequate prompts can lead to ambiguous or inaccurate responses [13].</li>
  <li>Prompts can be used for tasks like text summarization, information extraction, Q&A, text classification, language/code translation, code generation, and code documentation or reasoning [16].</li>
  <li>Prompts might need optimization for the specific model being used (e.g., Gemini, GPT, Claude, Gemma, LLaMA) [16].</li>
  <li>Types of prompting techniques:
    <ul>
      <li><strong>Zero-shot prompting:</strong> The simplest type, providing only a task description and input text without examples [17]. Example: Classifying movie reviews [17, 18].</li>
      <li><strong>System prompting:</strong> Provides an additional task to the system, useful for generating output that meets specific requirements, like returning output in uppercase or JSON format [19, 20].</li>
      <li><strong>Defining a role perspective:</strong> Giving the model a role (e.g., travel guide) to influence tone, style, and expertise for better quality and relevance [21, 22]. Styles include Descriptive, Formal, Humorous, Persuasive, etc. [22].</li>
      <li><strong>Chain of Thought (CoT) prompting:</strong> Involves asking the model to "think step by step" to improve reasoning [12, 23-26].</li>
      <li><strong>ReAct (Reasoning and Acting):</strong> Synergizes reasoning and acting in language models [26-29]. It makes a chain of searches and uses results as observations for the next step [27, 28]. Implementing ReAct requires resending previous prompts/responses and setting up the model with examples/instructions [28].</li>
      <li><strong>Automatic Prompt Engineering (APE):</strong> A method to automate the process of writing prompts [30]. A model is prompted to generate more prompts, which are then evaluated and potentially altered iteratively [30]. This can alleviate human input and enhance model performance [30].</li>
    </ul>
  </li>
  <li>Vertex AI Studio provides a playground for testing prompts [17, 31].</li>
  <li>Vertex AI offers configuration access (like temperature) when prompting the model directly or via API, compared to a consumer chatbot like Gemini [15].</li>
  <li>LLMs can help with code-related tasks like generating Bash scripts [31, 32], explaining code [33], and translating code between languages (e.g., Bash to Python) [34, 35]. It is essential to read and test generated code as LLMs can repeat training data and can't reason [32].</li>
  <li>Documenting prompting work, including prompt versions, results (OK/NOT OK/SOMETIMES OK), and feedback, is recommended [36]. Using a Google Sheet or saving prompts in Vertex AI Studio can help manage this [36].</li>
</ul>

<h2>Embeddings & Vector Stores</h2>
<ul>
  <li><strong>Embeddings</strong> transform diverse data (images, text, audio, etc.) into a unified vector representation [37].</li>
  <li>Deep neural network models for embeddings perform better than models using bag-of-words paradigms [38]. Embeddings can differ based on context, unlike bag-of-words [38].</li>
  <li>Embeddings can be created for text [39, 40], images, and multimodal data [41, 42].</li>
  <li>For structured data, an embedding model needs to be created specifically for the application [42].</li>
  <li>Tools for creating embeddings include Google Vertex APIs (e.g., text-embedding-005 for health-related data, text-embedding-gecko@004), TensorFlow-hub (e.g., Sentence t5, BERT models) [38-40, 43, 44], and Google's BigQuery [38, 41].</li>
  <li>Vertex AI allows customization or fine-tuning of text embedding models [44]. Existing models from tfhub can also be loaded and fine-tuned [44].</li>
  <li><strong>Vector databases</strong> are specialized systems for managing multi-dimensional data, enabling efficient searching and retrieval within high-dimensional spaces [45]. Data is represented as vectors capturing its semantic meaning [45].</li>
  <li>Vertex AI offers three solutions for storing and serving embeddings at scale [45].</li>
  <li>A key application is <strong>Retrieval Augmented Generation (RAG)</strong>, where embeddings and vector stores are used to retrieve relevant information to ground LLM responses [37, 46, 47]. This can be implemented scalably using Vertex AI LLM text embeddings and Vertex AI Vector Search with libraries like LangChain [47]. Vertex AI Search is a fully managed service for search and RAG built on Google Search technologies [48, 49].</li>
  <li>Vector stores can be initialized as retrievers to find relevant text chunks based on a query [46].</li>
</ul>

<h2>Agents</h2>
<ul>
  <li>An <strong>agent</strong> is a program that extends beyond the standalone capabilities of a Generative AI model [50].</li>
  <li>Agents can use tools to access real-time information or suggest real-world actions [50, 51].</li>
  <li>They require the ability to plan and execute tasks in a self-directed fashion, combining reasoning, logic, and access to external information [50].</li>
  <li>The core of an agent system comprises three fundamental elements: a <strong>Foundation Model</strong>, <strong>Instructions</strong>, and a <strong>Tool</strong> [52].</li>
  <li>The Foundation Model provides reasoning and language processing [52]. Instructions are the guiding directives or goals [52]. The Tool consists of descriptions of available functions and parameters, which the model uses to reason about which tool to use and how [52].</li>
  <li>Examples of tools include database retrieval tools, API calls (e.g., to send an email or complete a transaction) [50], Extensions (like Code Interpreter to run Python code from natural language) [53], Functions (custom Python methods) [54], SerpAPI (for Google Search) [55], and Google Places API [55].</li>
  <li>Google provides out-of-the-box Extensions to simplify tool usage [53].</li>
  <li>Defining a role perspective for an AI model improves output quality, relevance, and effectiveness [22].</li>
  <li>Enhancing model performance with targeted learning is crucial for effective tool selection [56].</li>
  <li>Building production-grade agent applications requires integrating with tools like user interfaces, evaluation frameworks, and continuous improvement mechanisms [57]. Google's Vertex AI platform offers a fully managed environment for this [57].</li>
  <li>On Vertex AI, developers can define agent elements (goals, task instructions, tools, sub-agents, examples) using a natural language interface [57]. The platform includes development tools for testing, evaluation, performance measurement, debugging, and improving agents [57].</li>
  <li>The <strong>Agent Lifecycle</strong> involves building, deploying, and managing agent-based systems, considering tooling, observability, safety, and continuous improvement [58].</li>
  <li>Productionizing agents involves CI/CD pipelines and a centralized Tool Registry for a robust deployment process [59].</li>
  <li><strong>Multi-agent systems</strong> involve a team of specialized agents collaborating to solve complex problems [60, 61]. The automotive domain is a compelling case study, demonstrating coordination patterns (hierarchical, collaborative, peer-to-peer) [60].</li>
  <li>Google's AI co-scientist is an example of a multi-agent LLM system for scientific research, using a "generate, debate, and evolve" approach [61].</li>
  <li>Examples of specialized agents in a multi-agent system include a Car Manual Agent (using RAG for car-related questions) [62] and a Rephraser Agent (adapting tone, style, and presentation) [63].</li>
  <li>Tools for agents mentioned include Vertex AI Search [48, 49], search builder APIs [48], RAG Engine [48], NotebookLM Enterprise (a research and learning tool) [64-66], Vertex Eval Service [66], Cloud Observability [66], and Vertex AI Agent Builder [67].</li>
</ul>

<h2>Solving Domain-Specific Problems using LLMs</h2>
<ul>
  <li>LLMs can be applied to <strong>domain-specific problems</strong> in various areas of expertise [68].</li>
  <li>This presents challenges related to specialized data, technical language, and sensitive use cases [68].</li>
  <li>Examples discussed are SecLM for cybersecurity and Med-PaLM for healthcare [68].</li>
  <li>Specialized Gen AI can help address challenges like evolving threats, operational toil, and talent shortages in security by automating repetitive tasks and providing knowledge access [68].</li>
  <li>Med-PaLM is a large language model focused on medical question answering [69, 70].</li>
</ul>

<h2>Operationalizing Generative AI on Vertex AI</h2>
<ul>
  <li>Operationalizing Gen AI systems involves intricate workflows chaining models, APIs, and data sources [58].</li>
  <li>Existing MLOps and DevOps practices apply to MLOps for Gen AI, including governing the data, tuned model, and code lifecycles [71].</li>
  <li>The Vertex AI platform is designed to address the unique demands of foundation models and Gen AI applications, offering a comprehensive MLOps platform [72].</li>
  <li>Vertex AI Model Garden provides a curated collection of over 150 ML and Gen AI models from Google, partners, and open source, simplifying discovery, customization, and deployment [73, 74]. This includes Google's Gemini family [75], Imagen [75], Codey [75], and pre-trained APIs [74], as well as open source models like Gemma, Llama, Falcon, BERT, T-5 FLAN, ViT, and EfficientNet [74].</li>
  <li>Vertex AI Studio is a console-driven entry point to access and leverage Vertex AI's Gen AI services [76]. It allows exploration and experimentation with Google's first-party models, testing prompts with different parameters, and adapting models through techniques like supervised fine-tuning (SFT), reinforcement learning tuning, and Distillation [76].</li>
  <li>Vertex AI offers a comprehensive ecosystem for augmenting LLMs to address factual grounding and hallucination, including RAG and agent-based approaches [77].</li>
  <li>Vertex AI Search is a fully managed search and RAG provider built on Google Search technologies [48, 49]. It allows grounding agents by indexing diverse data sources like BigQuery, Cloud SQL, website content, cloud storage, and third-party APIs [49]. It abstracts complexities like semantic search, hybrid search, and embedding models [49].</li>
  <li>Vertex AI provides solutions for storing and serving embeddings at scale, catering to diverse use cases and user profiles [45].</li>
  <li>Vertex AI endpoints offer features for deploying and managing models, including citation checkers (providing quotes for websites and code repositories) [78], watermarking for AI-generated images (using SynthID) [79], and content moderation and bias detection tools to mitigate risks [79].</li>
  <li>Productionizing Gen AI solutions requires a well-orchestrated interplay of diverse individuals and standardized processes [59].</li>
</ul>
