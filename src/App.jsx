import { useMemo, useRef, useState } from "react";

const MODEL_ID = "onnx-community/Llama-3.2-1B-Instruct";
const DEFAULT_SYSTEM_PROMPT = "You are a concise, helpful assistant.";

let pipelineModulePromise;

async function getPipelineModule() {
  if (!pipelineModulePromise) {
    pipelineModulePromise = import("@huggingface/transformers");
  }

  return pipelineModulePromise;
}

export default function App() {
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content: "Hi. Load the model, then send a message to start chatting.",
    },
  ]);
  const [input, setInput] = useState("");
  const [systemPrompt, setSystemPrompt] = useState(DEFAULT_SYSTEM_PROMPT);
  const [status, setStatus] = useState("Model not loaded yet.");
  const [error, setError] = useState("");
  const [isLoadingModel, setIsLoadingModel] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);

  const generatorRef = useRef(null);
  const generatorPromiseRef = useRef(null);
  const webgpuAvailable = useMemo(() => typeof navigator !== "undefined" && "gpu" in navigator, []);

  async function getGenerator() {
    if (generatorRef.current) {
      return generatorRef.current;
    }

    if (generatorPromiseRef.current) {
      return generatorPromiseRef.current;
    }

    setError("");
    setIsLoadingModel(true);
    setStatus(`Loading ${MODEL_ID}… the first run downloads the model and may take a while.`);

    generatorPromiseRef.current = getPipelineModule()
      .then(async ({ env, pipeline }) => {
        env.allowLocalModels = false;
        const generator = await pipeline("text-generation", MODEL_ID);
        generatorRef.current = generator;
        setStatus("Model ready.");
        return generator;
      })
      .catch((loadError) => {
        generatorPromiseRef.current = null;
        throw loadError;
      })
      .finally(() => {
        setIsLoadingModel(false);
      });

    return generatorPromiseRef.current;
  }

  async function loadModel() {
    try {
      await getGenerator();
    } catch (loadError) {
      console.error(loadError);
      setError(loadError?.message || "Failed to load the model.");
      setStatus("Model load failed.");
    }
  }

  function clearChat() {
    setMessages([
      {
        role: "assistant",
        content: "Chat cleared. Send a message when you are ready.",
      },
    ]);
    setError("");
  }

  async function sendMessage() {
    const text = input.trim();
    if (!text || isGenerating || isLoadingModel) {
      return;
    }

    setError("");
    setInput("");

    const userMessage = { role: "user", content: text };
    const placeholderMessage = { role: "assistant", content: "Thinking…", pending: true };
    const visibleHistory = [...messages, userMessage]
      .filter((message) => !message.pending)
      .map(({ role, content }) => ({ role, content }));
    const conversation = [
      {
        role: "system",
        content: systemPrompt.trim() || DEFAULT_SYSTEM_PROMPT,
      },
      ...visibleHistory,
    ];

    setMessages((previousMessages) => [...previousMessages, userMessage, placeholderMessage]);
    setIsGenerating(true);
    setStatus("Generating response…");

    try {
      const generator = await getGenerator();
      const output = await generator(conversation, {
        do_sample: true,
        max_new_tokens: 256,
        repetition_penalty: 1.05,
        temperature: 0.7,
        top_p: 0.9,
      });
      const generated = output?.[0]?.generated_text;
      const assistantText = Array.isArray(generated)
        ? generated.at(-1)?.content || ""
        : typeof generated === "string"
          ? generated
          : "";

      setMessages((previousMessages) => {
        const nextMessages = [...previousMessages];
        const pendingIndex = nextMessages.findLastIndex((message) => message.pending);

        if (pendingIndex >= 0) {
          nextMessages[pendingIndex] = {
            role: "assistant",
            content: assistantText || "No response returned.",
          };
        }

        return nextMessages;
      });
      setStatus("Ready.");
    } catch (generationError) {
      console.error(generationError);
      const message = generationError?.message || "Generation failed.";
      setError(message);
      setMessages((previousMessages) => {
        const nextMessages = [...previousMessages];
        const pendingIndex = nextMessages.findLastIndex((message) => message.pending);

        if (pendingIndex >= 0) {
          nextMessages[pendingIndex] = {
            role: "assistant",
            content: `Error: ${message}`,
          };
        }

        return nextMessages;
      });
      setStatus("Generation failed.");
    } finally {
      setIsGenerating(false);
    }
  }

  function handleComposerKeyDown(event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void sendMessage();
    }
  }

  return (
    <main className="app-shell">
      <section className="panel">
        <div className="panel-header">
          <div>
            <p className="eyebrow">GitHub Pages-ready SPA</p>
            <h1>Transformers.js chat</h1>
          </div>
          <span className="status-pill">{status}</span>
        </div>

        <p className="panel-copy">
          Browser-only demo for <strong>{MODEL_ID}</strong>. The first model download is large and may
          take a while.
        </p>

        <dl className="meta-grid">
          <div>
            <dt>WebGPU available</dt>
            <dd>{webgpuAvailable ? "Yes" : "No"}</dd>
          </div>
          <div>
            <dt>Model source</dt>
            <dd>Hugging Face Hub</dd>
          </div>
          <div>
            <dt>Runtime</dt>
            <dd>Transformers.js in the browser</dd>
          </div>
        </dl>

        <label className="field">
          <span>System prompt</span>
          <textarea
            className="text-area"
            value={systemPrompt}
            onChange={(event) => setSystemPrompt(event.target.value)}
          />
        </label>

        {error ? <div className="error-box">{error}</div> : null}

        <div className="button-row">
          <button
            type="button"
            className="button primary"
            onClick={loadModel}
            disabled={isLoadingModel || Boolean(generatorRef.current)}
          >
            {isLoadingModel ? "Loading model…" : generatorRef.current ? "Model loaded" : "Load model"}
          </button>
          <button type="button" className="button secondary" onClick={clearChat}>
            Clear chat
          </button>
        </div>
      </section>

      <section className="panel">
        <div className="panel-header">
          <div>
            <p className="eyebrow">Conversation</p>
            <h2>Chat with the local model</h2>
          </div>
        </div>

        <div className="message-list" aria-live="polite">
          {messages.map((message, index) => (
            <article key={`${message.role}-${index}`} className={`message ${message.role}`}>
              <header>{message.role}</header>
              <p>{message.content}</p>
            </article>
          ))}
        </div>

        <label className="field">
          <span>Message</span>
          <textarea
            className="composer"
            value={input}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={handleComposerKeyDown}
            placeholder="Type your message and press Enter to send."
            disabled={isLoadingModel || isGenerating}
          />
        </label>

        <div className="button-row">
          <button
            type="button"
            className="button primary"
            onClick={() => void sendMessage()}
            disabled={!input.trim() || isLoadingModel || isGenerating}
          >
            {isGenerating ? "Generating…" : "Send"}
          </button>
        </div>
      </section>
    </main>
  );
}
