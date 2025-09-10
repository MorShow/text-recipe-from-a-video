import React from "react";
import { useNavigate } from "react-router-dom";

const recipes = [
  { id: "spaghetti", title: "Spaghetti Bolognese", color: "bg-red-400" },
  { id: "curry", title: "Chicken Curry", color: "bg-green-400" },
  { id: "stirfry", title: "Vegetable Stir Fry", color: "bg-blue-400" },
  { id: "pancakes", title: "Pancakes", color: "bg-yellow-400" },
];

const Home = () => {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-white p-8">
      <h1 className="text-4xl font-bold text-gray-800 mb-8 text-center">
        🍽️ My Recipes
      </h1>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 max-w-4xl mx-auto">
        {recipes.map((recipe) => (
          <div
            key={recipe.id}
            className={`cursor-pointer rounded-2xl shadow-lg p-6 text-white text-center font-semibold text-xl hover:scale-105 transform transition ${recipe.color}`}
            onClick={() => navigate(`/recipe/${recipe.id}`)}
          >
            {recipe.title}
          </div>
        ))}
      </div>
    </div>
  );
};

export default Home;