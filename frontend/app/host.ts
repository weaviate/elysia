const host =
  process.env.NODE_ENV === "development"
    ? "localhost:8000"
    : "elysia-backend.onrender.com";

export default host;
