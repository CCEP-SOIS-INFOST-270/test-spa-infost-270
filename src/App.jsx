import { useMemo, useRef, useState } from "react";

const MODEL_ID = "onnx-community/Llama-3.2-1B-Instruct";
const MODEL_TASK = "text-generation";
const MODEL_DTYPE = "q4";
const DEFAULT_SYSTEM_PROMPT = "You are a concise, helpful assistant.";

let pipelineModulePromise;

async function getPipelineModule() {
  if (!pipelineModulePromise) {
    pipelineModulePromise = import("@huggingface/transformers");
  }

  return pipelineModulePromise;
}

function formatBytes(value) {
  if (!Number.isFinite(value) || value < 0) {
    return null;
  }

  if (value === 0) {
    return "0 B";
  }

  const units = ["B", "KB", "MB", "GB"];
  const exponent = Math.min(Math.floor(Math.log(value) / Math.log(1024)), units.length - 1);
  const amount = value / 1024 ** exponent;

  return `${amount.toFixed(amount >= 10 || exponent === 0 ? 0 : 1)} ${units[exponent]}`;
}

function getFileLabel(file) {
  return file ? file.split("/").at(-1) || file : "model files";
}

function handleCacheStatusError(error) {
  console.warn("Unable to inspect model cache status.", error);
  return false;
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
  const [isModelCached, setIsModelCached] = useState(false);
  const [loadProgress, setLoadProgress] = useState({
    visible: false,
    label: "Load the q4 model to start the download.",
    percent: 0,
    loaded: 0,
    total: 0,
  });

  const generatorRef = useRef(null);
  const generatorPromiseRef = useRef(null);
  const webgpuAvailable = useMemo(() => typeof navigator !== "undefined" && Boolean(navigator.gpu), []);
  const browserCacheAvailable = useMemo(() => typeof caches !== "undefined", []);

  function handleModelProgress(progressInfo) {
    switch (progressInfo.status) {
      case "initiate":
        setLoadProgress((previousProgress) => ({
          ...previousProgress,
          visible: true,
          label: `Preparing ${getFileLabel(progressInfo.file)}…`,
        }));
        break;
      case "download":
        setLoadProgress((previousProgress) => ({
          ...previousProgress,
          visible: true,
          label: `Downloading ${getFileLabel(progressInfo.file)}…`,
        }));
        break;
      case "progress_total": {
        const percent = Math.max(0, Math.min(100, Math.round(progressInfo.progress)));
        const loaded = formatBytes(progressInfo.loaded);
        const total = formatBytes(progressInfo.total);

        setLoadProgress({
          visible: true,
          label:
            loaded && total
              ? `Downloading and loading model files (${loaded} of ${total})…`
              : "Downloading and loading model files…",
          percent,
          loaded: progressInfo.loaded,
          total: progressInfo.total,
        });
        setStatus(`Loading ${MODEL_ID} (${MODEL_DTYPE})… ${percent}%`);
        break;
      }
      case "done":
        setLoadProgress((previousProgress) => ({
          ...previousProgress,
          visible: true,
          label: `Loaded ${getFileLabel(progressInfo.file)}.`,
        }));
        break;
      case "ready":
        setLoadProgress((previousProgress) => ({
          ...previousProgress,
          visible: true,
          label: "Finishing model setup…",
          percent: 100,
        }));
        break;
      default:
        break;
    }
  }

  async function getGenerator() {
    if (generatorRef.current) {
      return generatorRef.current;
    }

    if (generatorPromiseRef.current) {
      return generatorPromiseRef.current;
    }

    setError("");
    setIsModelCached(false);
    setIsLoadingModel(true);
    setLoadProgress({
      visible: true,
      label: browserCacheAvailable
        ? "Checking the browser cache and preparing the q4 model…"
        : "Preparing the q4 model…",
      percent: 0,
      loaded: 0,
      total: 0,
    });
    setStatus(`Loading ${MODEL_ID} (${MODEL_DTYPE})… the first run downloads the model and may take a while.`);

    generatorPromiseRef.current = getPipelineModule()
      .then(async ({ ModelRegistry, env, pipeline }) => {
        env.useBrowserCache = browserCacheAvailable;

        const cacheOptions = { dtype: MODEL_DTYPE };
        const wasCachedBeforeLoad = browserCacheAvailable
          ? await ModelRegistry.is_pipeline_cached(MODEL_TASK, MODEL_ID, cacheOptions).catch(
              handleCacheStatusError,
            )
          : false;

        if (wasCachedBeforeLoad) {
          setIsModelCached(true);
          setLoadProgress((previousProgress) => ({
            ...previousProgress,
            visible: true,
            label: "Using locally cached q4 model files when available…",
          }));
        }

        const generator = await pipeline(MODEL_TASK, MODEL_ID, {
          dtype: MODEL_DTYPE,
          progress_callback: handleModelProgress,
        });
        generatorRef.current = generator;

        const cachedAfterLoad = browserCacheAvailable
          ? await ModelRegistry.is_pipeline_cached(MODEL_TASK, MODEL_ID, cacheOptions).catch(
              handleCacheStatusError,
            )
          : false;

        setIsModelCached(cachedAfterLoad);
        setLoadProgress((previousProgress) => ({
          ...previousProgress,
          visible: true,
          label: cachedAfterLoad
            ? wasCachedBeforeLoad
              ? "Model ready from local browser cache."
              : "Model ready and cached locally in this browser."
            : "Model ready.",
          percent: 100,
        }));
        setStatus(
          cachedAfterLoad
            ? wasCachedBeforeLoad
              ? "Model ready from local cache."
              : "Model ready and cached locally."
            : "Model ready.",
        );
        return generator;
      })
      .catch((loadError) => {
        generatorPromiseRef.current = null;
        setLoadProgress({
          visible: true,
          label: "Model load failed.",
          percent: 0,
          loaded: 0,
          total: 0,
        });
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

  function handleClearChat() {
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
          Browser-only demo for <strong>{MODEL_ID}</strong> using the <strong>{MODEL_DTYPE}</strong>{" "}
          quantized weights. The first download is large, but browser caching keeps the model local
          for faster reloads when storage is available.
        </p>

        <dl className="meta-grid">
          <div>
            <dt>WebGPU available</dt>
            <dd>{webgpuAvailable ? "Yes" : "No"}</dd>
          </div>
          <div>
            <dt>Model source</dt>
            <dd>Hugging Face Hub + browser cache</dd>
          </div>
          <div>
            <dt>Runtime</dt>
            <dd>Transformers.js in the browser</dd>
          </div>
          <div>
            <dt>Model variant</dt>
            <dd>{MODEL_DTYPE}</dd>
          </div>
          <div>
            <dt>Local cache</dt>
            <dd>
              {browserCacheAvailable
                ? isModelCached
                  ? "Ready in browser cache"
                  : "Will cache after first load"
                : "Unavailable"}
            </dd>
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
            {isLoadingModel
              ? "Loading q4 model…"
              : generatorRef.current
                ? "Model loaded"
                : "Load q4 model"}
          </button>
          <button type="button" className="button secondary" onClick={handleClearChat}>
            Clear chat
          </button>
        </div>

        {loadProgress.visible ? (
          <div className="progress-card">
            <div className="progress-meta">
              <span>{loadProgress.label}</span>
              <strong>{loadProgress.percent}%</strong>
            </div>
            <div
              className="progress-track"
              role="progressbar"
              aria-label="Model load progress"
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={loadProgress.percent}
            >
              <div className="progress-fill" style={{ width: `${loadProgress.percent}%` }} />
            </div>
            <p className="progress-copy">
              {loadProgress.total > 0
                ? `${formatBytes(loadProgress.loaded)} of ${formatBytes(loadProgress.total)} loaded`
                : browserCacheAvailable
                  ? isModelCached
                    ? "Cached model files can be reused after refresh."
                    : "Downloaded model files will be stored in the browser cache."
                  : "This browser environment does not expose the cache API."}
            </p>
          </div>
        ) : null}
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
