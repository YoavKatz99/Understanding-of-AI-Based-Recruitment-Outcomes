import React, { useState } from "react";

export default function ToolSelector() {
  const [tool, setTool] = useState("shap");
  const [file, setFile] = useState(null);
  const [score, setScore] = useState(null);
  const [error, setError] = useState(null);
  const [resultFile, setResultFile] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setScore(null);
    setResultFile(null);
    setError(null);
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

    try {
      const response = await fetch("http://127.0.0.1:5000/explain", {
        method: "POST",
        body: formData,
      });

      const result = await response.json();
      console.log("Server result:", result);

      if (response.ok && typeof result.prediction === "number") {
        setScore(result.prediction);
        if (tool === "shap") {
          setResultFile("shap_importance.png");
        } else if (tool === "lime") {
          setResultFile("lime_explanation.html");
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
            <option value="lime">LIME</option>
          </select>

          <label className="cursor-pointer bg-indigo-600 text-white px-4 py-2 rounded-md hover:bg-indigo-700">
            Upload CV
            <input
              type="file"
              onChange={handleFileChange}
              className="hidden"
            />
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
          <p className="text-center text-red-600 text-lg font-medium mb-6">❌ Error: {error}</p>
        )}

        {score !== null && (
          <div className="bg-white rounded-2xl shadow-xl p-10">
            <div className="text-center">
              <h2 className="text-2xl font-bold text-gray-800">Match Score</h2>
              <p className="text-6xl font-extrabold text-indigo-600 mt-2">{score.toFixed(2)}%</p>
              <p className="mt-4 text-gray-600">
                This score reflects how well the resume aligns with the job requirements based on extracted skills.
              </p>
            </div>

            <div className="mt-12 grid md:grid-cols-2 gap-8">
              {tool === "shap" && resultFile && (
                <div>
                  <h3 className="text-xl font-semibold text-gray-700 mb-3">SHAP Feature Importance</h3>
                  <img
                    src={`http://127.0.0.1:5000/outputs/${resultFile}`}
                    alt="SHAP Importance"
                    className="rounded-xl border w-full"
                  />
                </div>
              )}

              {tool === "lime" && resultFile && (
                <div>
                  <h3 className="text-xl font-semibold text-gray-700 mb-3">LIME Explanation</h3>
                  <iframe
                    src={`http://127.0.0.1:5000/outputs/${resultFile}`}
                    className="w-full h-[600px] border rounded-xl"
                    title="LIME Explanation"
                  ></iframe>
                </div>
              )}

              <div className="bg-gray-50 rounded-xl p-6 shadow-inner">
                <h4 className="text-lg font-semibold text-indigo-700 mb-2">Interpretation Tips</h4>
                <ul className="text-gray-600 list-disc pl-6 space-y-1 text-sm">
                  <li><strong>SHAP</strong>: shows which features (skills) pushed the score up or down.</li>
                  <li><strong>LIME</strong>: visualizes how individual features contributed to this specific prediction.</li>
                  <li>Negative values reduce match score; positive values increase it.</li>
                  <li>Use insights to improve the resume (e.g., add missing keywords).</li>
                </ul>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
