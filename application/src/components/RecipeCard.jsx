import React from "react";
import { useNavigate } from "react-router-dom";

const RecipeCard = ({ title, image, description }) => {
  const navigate = useNavigate();

  return (
    <div className="border rounded-lg overflow-hidden shadow-lg hover:shadow-xl transition p-4 flex flex-col">
      <img src={image} alt={title} className="h-40 w-full object-cover mb-4" />
      <h2 className="font-bold text-xl mb-2">{title}</h2>
      <p className="text-sm mb-4">{description}</p>
      <button
        onClick={() => navigate(`/recipe/${title}`)}
        className="mt-auto bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
      >
        View Recipe
      </button>
    </div>
  );
};

export default RecipeCard;