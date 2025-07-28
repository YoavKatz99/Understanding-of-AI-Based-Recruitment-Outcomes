import React, { useState, useEffect } from "react";

export default function ToolSelector() {
  useEffect(() => {
    const handleResize = (event) => {
      if (event.origin.includes("127.0.0.1")) {
        const iframe = document.querySelector("iframe");
        if (iframe && event.data.height) {
          iframe.style.height = event.data.height + "px";
        }
      }
    };

    window.addEventListener("message", handleResize);
    return () => window.removeEventListener("message", handleResize);
  }, []);

  const [tool, setTool] = useState("shap");
  const [file, setFile] = useState(null);
  const [score, setScore] = useState(null);
  const [error, setError] = useState(null);
  const [resultFile, setResultFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [outputText, setOutputText] = useState(null);
  const [highlightedWords, setHighlightedWords] = useState([]);
  
  // Define API base at component level so it's available everywhere
  const apiBase = process.env.REACT_APP_API_BASE_URL || 
                 "https://understanding-of-ai-based-recruitment.onrender.com";

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setScore(null);
    setResultFile(null);
    setError(null);
    setOutputText(null);
  };

  const handleSubmit = async () => {
    if (!file) {
      alert("Please select a file.");
      return;
    }

    setLoading(true);
    const formData = new FormData();
    formData.append("file", file);
    formData.append("tool", tool);

    let endpoint = `${apiBase}/explain`;
    if (tool === "dice") {
      endpoint = `${apiBase}/explain_carla`;
    } else if (tool === "lime_text") {
      endpoint = `${apiBase}/explain_lime_text`;
    }

    try {
      const response = await fetch(endpoint, {
        method: "POST",
        body: formData,
      });

      const result = await response.json();

      if (response.ok && typeof result.prediction === "number") {
        setScore(result.prediction);

        if (tool === "shap") {
          setResultFile("shap_importance.png");
        } else if (tool === "lime_text") {
          setResultFile("lime_text_explanation.html");
          setHighlightedWords(result.highlighted_words || []);
        } else if (tool === "dice") {
          setResultFile(result.output_file);
          if (result.output_text) {
            setOutputText(result.output_text);
          }
        }

        setError(null);
      } else {
        setError(result.error || "Unexpected error");
      }
    } catch (err) {
      console.error("Fetch error:", err);
      setError("Failed to connect to backend.");
    } finally {
      setLoading(false);
    }
  };

  const renderDiceOutput = (text) => {
    if (!text) return null;
    return text.split("\n").map((line, index) => {
      if (line.match(/^[A-Z =]+$/)) {
        return (
          <h3 key={index} className="font-bold text-lg mt-4 mb-2">
            {line}
          </h3>
        );
      } else if (line.trim() === "") {
        return <br key={index} />;
      } else if (line.match(/^Scenario \d+:/)) {
        return (
          <h4 key={index} className="font-semibold text-md mt-3 mb-1">
            {line}
          </h4>
        );
      } else if (line.trim().startsWith("- ")) {
        return (
          <li key={index} className="ml-6 list-disc">
            {line.substring(2)}
          </li>
        );
      } else {
        return (
          <p key={index} className="my-1">
            {line}
          </p>
        );
      }
    });
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-100 to-white py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-5xl mx-auto">
        <h1 className="text-5xl font-extrabold text-center text-indigo-800 mb-12 tracking-tight">
          Resume Insight Analyzer
        </h1>

        <div className="flex flex-col md:flex-row justify-center items-center gap-4 mb-12">
          <select
            className="border border-gray-300 rounded-md px-4 py-2 shadow-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
            value={tool}
            onChange={(e) => setTool(e.target.value)}
          >
            <option value="shap">SHAP</option>
            <option value="lime_text">LIME (Text)</option>
            <option value="dice">DiCE</option>
          </select>

          <label className="cursor-pointer bg-indigo-600 text-white px-4 py-2 rounded-md hover:bg-indigo-700">
            Upload CV
            <input type="file" onChange={handleFileChange} className="hidden" />
          </label>

          <button
            onClick={handleSubmit}
            className="bg-indigo-600 text-white px-6 py-2 rounded-md font-semibold shadow-md hover:bg-indigo-700 transition disabled:opacity-50"
            disabled={loading}
          >
            {loading ? "Analyzing..." : "Analyze"}
          </button>
        </div>

        {error && (
          <p className="text-center text-red-600 text-lg font-medium mb-6">
            ❌ Error: {error}
          </p>
        )}

        <div className="bg-gray-50 rounded-xl p-6 shadow-inner">
          <h4 className="text-lg font-semibold text-indigo-700 mb-2">
            Interpretation Tips
          </h4>
          <ul className="text-gray-600 list-disc pl-6 space-y-1 text-sm">
            <li>
              <strong>SHAP</strong>: shows which features (skills) pushed
              the score up or down.
            </li>
            <li>
              <strong>LIME</strong>: visualizes how individual features
              contributed to this specific prediction.
            </li>
            <li>
              <strong>DiCE</strong>: suggests which skills to add to
              improve your score.
            </li>
            <li>
              Use insights to improve the resume (e.g., add missing
              keywords).
            </li>
          </ul>
        </div>

        {score !== null && (
          <div className="bg-white rounded-2xl shadow-xl p-10 mt-8">
            <div className="text-center">
              <h2 className="text-2xl font-bold text-gray-800">Match Score</h2>
              <p className="text-6xl font-extrabold text-indigo-600 mt-2">
                {score.toFixed(2)}%
              </p>
              <p className="mt-4 text-gray-600">
                This score reflects how well the resume aligns with the job
                requirements based on extracted skills.
              </p>
            </div>

            <div className="mt-12 grid md:grid-cols-1 gap-8">
              {tool === "shap" && resultFile && (
                <div>
                  <h3 className="text-xl font-semibold text-gray-700 mb-3">
                    SHAP Feature Importance
                  </h3>
                  <p className="text-sm text-gray-600 mb-4">
                    This is global feature importance plot, where the global importance of each feature is taken to be the mean absolute value for that feature over all the given samples.
                  </p>
                  <img
                    src={`${apiBase}/outputs/${resultFile}`}
                    alt="SHAP Importance"
                    className="rounded-xl border w-full"
                  />
                </div>
              )}

              {tool === "lime_text" && resultFile && (
                <div>
                  <h3 className="text-xl font-semibold text-gray-700 mb-3">
                    LIME Explanation
                  </h3>
                  <p className="text-sm text-gray-600 mb-4">
                    This explanation highlights key words in your resume and shows whether they positively or negatively affected your match score. Words are color-coded: orange means a positive effect, blue means negative.
                  </p>
                  <div className="w-full">
                    <iframe
                      src={`${apiBase}/outputs/${resultFile}`}
                      title="LIME Explanation"
                      className="w-full h-[600px] border rounded-xl shadow"
                    />
                  </div>
                </div>
              )}

              {tool === "dice" && (
                <div>
                  <h3 className="text-xl font-semibold text-gray-700 mb-3">
                    DiCE Explanation
                  </h3>
                  <p className="text-sm text-gray-600 mb-4">
                    These suggestions show how your resume could be changed to receive a higher score. Each scenario lists specific changes (like adding skills or modifying terms) that would improve your match percentage.
                  </p>
                  {outputText ? (
                    <div className="p-4 border rounded bg-white text-black">
                      <div className="font-mono whitespace-pre-wrap">
                        {renderDiceOutput(outputText)}
                      </div>
                    </div>
                  ) : resultFile ? (
                    <div>
                      <p>Download the explanation:</p>
                      <a
                        href={`${apiBase}/outputs/${resultFile}`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-blue-600 underline"
                      >
                        View explanation file
                      </a>
                    </div>
                  ) : null}
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}