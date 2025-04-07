import React, { useState } from "react";

export default function ToolSelector() {
  const [tool, setTool] = useState("shap");
  const [file, setFile] = useState(null);
  const [score, setScore] = useState(null);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleSubmit = async () => {
    if (!file) {
      alert("Please select a file.");
      return;
    }

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
      

      if (response.ok) {
        setScore(result.match_score);
        setError(null);
      } else {
        setError(result.error);
        setScore(null);
      }
    } catch (err) {
      setError("Failed to connect to backend.");
      setScore(null);
    }
  };

  return (
    <div className="p-6">
      <h2 className="text-2xl font-bold mb-4">Choose Explanation Tool</h2>

      <select
        className="border rounded px-2 py-1 mr-2"
        value={tool}
        onChange={(e) => setTool(e.target.value)}
      >
        <option value="shap">SHAP</option>
        <option value="lime">LIME</option>
      </select>

      <input type="file" className="mr-2" onChange={handleFileChange} />

      <button
        onClick={handleSubmit}
        className="bg-white text-black border border-black px-4 py-2 rounded hover:bg-gray-100"
      >
        Submit
      </button>

      {typeof score === "number" && !isNaN(score) && (
        <p className="mt-4 text-lg font-semibold">
          Match Score: {score.toFixed(2)}%
        </p>
      )}

      {error && <p className="mt-4 text-red-600">Error: {error}</p>}
    </div>
  );
}