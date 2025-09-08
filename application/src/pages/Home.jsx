import React from "react";
import RecipeCard from "../components/RecipeCard";
import placeholder from "../assets/placeholder.jpg";

const recipes = [
  { title: "Spaghetti Bolognese", description: "Classic Italian pasta.", image: placeholder },
  { title: "Chicken Curry", description: "Spicy and creamy curry.", image: placeholder },
  { title: "Vegetable Stir Fry", description: "Quick and healthy.", image: placeholder },
  { title: "Pancakes", description: "Fluffy breakfast pancakes.", image: placeholder },
];

const Home = () => {
  return (
    <div className="container mx-auto p-8">
      <h1 className="text-3xl font-bold mb-8">Recipes</h1>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
        {recipes.map((recipe) => (
          <RecipeCard key={recipe.title} {...recipe} />
        ))}
      </div>
    </div>
  );
};

export default Home;