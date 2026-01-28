
try {
  await import("shiki");
  console.log("SUCCESS: Shiki can be imported in Node.js environment.");
} catch (error) {
  console.error("FAILURE: Shiki cannot be imported in Node.js environment.");
  console.error(error);
  process.exit(1);
}
