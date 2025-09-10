import React from "react";
import { useParams, Link } from "react-router-dom";

const Recipe = () => {
  const { recipeId } = useParams();

  return (
    <div className="min-h-screen bg-white p-8">
      <Link
        to="/"
        className="text-blue-600 underline hover:text-blue-800 text-sm"
      >
        ← Back to recipes
      </Link>
      <h1 className="text-3xl font-bold text-gray-800 mt-4">
        {recipeId.replace(/^\w/, (c) => c.toUpperCase())}
      </h1>
      <p className="mt-4 text-gray-600">
        This is where the recipe details will go.
      </p>
    </div>
  );
};

export default Recipe;