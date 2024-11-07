import { v4 as uuidv4 } from "uuid";

export const generateIdFromIp = async () => {
  try {
    const response = await fetch("https://api64.ipify.org?format=json");
    const data = await response.json();
    const ip = data.ip;
    const ipAsUuid = uuidv4(ip);
    return ipAsUuid;
  } catch (error) {
    console.error("Error fetching IP address:", error);
    return uuidv4(); // Fallback to a random UUID if IP fetch fails
  }
};
