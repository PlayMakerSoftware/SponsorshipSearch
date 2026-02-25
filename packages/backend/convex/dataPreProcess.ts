// packages/backend/convex/nflProcess.ts
import { action, mutation } from "./_generated/server";
import { v } from "convex/values";
import { api } from "./_generated/api";
import { Doc } from "./_generated/dataModel";


/**
 * Compute mean and sd, ignoring null; returns {mean, sd}
 */
function computeStats(nums: (number | null | undefined)[]): {
  mean: number;
  sd: number;
} {
  const filtered = nums.filter((n): n is number => typeof n === "number");

  if (filtered.length === 0) return { mean: 0, sd: 1 };

  const mean = filtered.reduce((a, b) => a + b, 0) / filtered.length;
  const variance =
    filtered.reduce((a, b) => a + (b - mean) ** 2, 0) / filtered.length;
  const sd = Math.sqrt(variance) || 1;

  return { mean, sd };
}

/**
 * Replace nulls with mean of column.
 */
// YUBI: should not need this function
function fillNull(value: number | null | undefined, mean: number): number {
  if (value === null || value === undefined) return mean;
  return value;
}

export const insertCleanRow = mutation({
    args: {
      row: v.any()
    },
    handler: async (ctx, { row }) => {
      await ctx.db.insert("NFL_seed_clean", row);
    }
});  


/**
 * MAIN MUTATION:
 * - Reads from NFL_seed
 * - Fills missing numeric values with column mean
 * - Normalizes numeric fields
 * - Embeds all string fields with Gemini
 * - Writes into NFL_clean
 */

// Define embed outside the action for better readability
async function embed(txt: string | undefined | null, apiKey: string): Promise<number[] | null> {
    if (!txt || txt.trim() === "") return null;
  
    const body = {
      model: "models/gemini-embedding-001",
      content: { parts: [{ text: txt }] }
    };
  
    const res = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent?key=${apiKey}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body)
      }
    );
  
    const json = await res.json();
    if (!res.ok) throw new Error(`Gemini Error: ${json.error?.message || res.statusText}`);
    
    return json.embedding.values;
  }
  
  export const buildCleanTable = action({
    args: {},
    handler: async (ctx) => {

        // 1. Access the environment variable via process.env
        const apiKey = process.env.GEMINI_API_KEY;

        // 2. Immediate validation: Stop early if the key is missing
        if (!apiKey) {
            throw new Error(
                "GEMINI_API_KEY is not set. Run 'npx convex env set GEMINI_API_KEY <your-key>'."
            );
        }

      const seed = await ctx.runQuery(api.NFL_seed.getAll, {});
      if (seed.length === 0) return "No rows in NFL_seed.";
  
      const valStats = computeStats(seed.map((r: Doc<"NFL_seed">) => r.valuation ?? null));
      const attStats = computeStats(seed.map((r: Doc<"NFL_seed">) => r.game_attendance ?? null));
      const igStats = computeStats(seed.map((r: Doc<"NFL_seed">) => r.instagram_followers ?? null));

      for (const row of seed) {
        // Parallelize the 4 embedding calls for THIS row
        // We use Promise.all to "await" all of them together
        const [regionEmb, leagueEmb, brandEmb, partnerEmb] = await Promise.all([
            row.region ? embed(row.region, apiKey) : Promise.resolve(null),
            row.league ? embed(row.league, apiKey) : Promise.resolve(null),
            row.brand_values ? embed(row.brand_values, apiKey) : Promise.resolve(null),
            row.current_partners ? embed(row.current_partners, apiKey) : Promise.resolve(null),
        ]);
  
        const cleanRow = {
          name: row.name,
          region: row.region,
          league: row.league,
          official_url: row.official_url,
  
          region_embedding: regionEmb,
          league_embedding: leagueEmb,
          brand_values_embedding: brandEmb,
          current_partners_embedding: partnerEmb,
  
          game_attendance_norm: (row.game_attendance != null) 
            ? (row.game_attendance - attStats.mean) / attStats.sd : null,
          valuation_norm: (row.valuation != null) 
            ? (row.valuation - valStats.mean) / valStats.sd : null,
          instagram_followers_norm: (row.instagram_followers != null) 
            ? (row.instagram_followers - igStats.mean) / igStats.sd : null,
        };
  
        await ctx.runMutation(api.dataPreProcess.insertCleanRow, { row: cleanRow });
      }
  
      return "NFL_seed_clean built successfully.";
    },
  });