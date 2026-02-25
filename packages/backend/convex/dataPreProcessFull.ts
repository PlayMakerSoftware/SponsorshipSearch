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
  max: number;
} {
  const filtered = nums.filter((n): n is number => typeof n === "number");

  if (filtered.length === 0) return { mean: 0, sd: 1, max: 0 };

  const mean = filtered.reduce((a, b) => a + b, 0) / filtered.length;
  const variance =
    filtered.reduce((a, b) => a + (b - mean) ** 2, 0) / filtered.length;
  const sd = Math.sqrt(variance) || 1;

  const max = Math.max(...filtered);

  return { mean, sd, max };
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
      await ctx.db.insert("All_Teams_Clean", row);
      
      // Update the count in tableCounts
      const countDoc = await ctx.db
        .query("tableCounts")
        .withIndex("by_table", (q) => q.eq("tableName", "All_Teams_Clean"))
        .unique();
      
      if (countDoc) {
        await ctx.db.patch(countDoc._id, { count: countDoc.count + 1 });
      } else {
        await ctx.db.insert("tableCounts", { tableName: "All_Teams_Clean", count: 1 });
      }
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

      const seed = await ctx.runQuery(api.All_Teams.getAll, {});
      if (seed.length === 0) return "No rows in All_Teams.";
  
      const attendance = computeStats(seed.map((r: Doc<"All_Teams">) => r.avg_game_attendance ?? null));
      const population = computeStats(seed.map((r: Doc<"All_Teams">) => r.city_population ?? null));
      const gdp = computeStats(seed.map((r: Doc<"All_Teams">) => r.metro_gdp ?? null));

    // put social media info on burner for now because Ibraheem hasn't been able to scrape it yet
      const x = computeStats(seed.map((r: Doc<"All_Teams">) => r.followers_x ?? null));
      const instagram = computeStats(seed.map((r: Doc<"All_Teams">) => r.followers_instagram ?? null));
      const facebook = computeStats(seed.map((r: Doc<"All_Teams">) => r.followers_facebook ?? null));
      const tiktok = computeStats(seed.map((r: Doc<"All_Teams">) => r.followers_tiktok ?? null));
      const youtube = computeStats(seed.map((r: Doc<"All_Teams">) => r.subscribers_youtube ?? null));
      const family_programs = computeStats(seed.map((r: Doc<"All_Teams">) => r.family_program_count ?? null));
      const ticketStats = computeStats(seed.map((r: Doc<"All_Teams">) => r.avg_ticket_price ?? null));
      const valuation = computeStats(seed.map((r: Doc<"All_Teams">) => r.franchise_value ?? null));
      const revenue = computeStats(seed.map((r: Doc<"All_Teams">) => r.annual_revenue ?? null));

      for (const row of seed) {
        // Parallelize the 48embedding calls for THIS row
        // We use Promise.all to "await" all of them together
        const [regionEmb, leagueEmb, valuesEmb, sponsorsEmb, familyProgramsEmb, communityProgramsEmb, partnersEmb] = await Promise.all([
            row.region ? embed(row.region, apiKey) : Promise.resolve(null),
            // Should I use league or category?
            row.league ? embed(row.league, apiKey) : Promise.resolve(null),
            row.mission_tags ? embed(row.mission_tags.join(" "), apiKey) : Promise.resolve(null),
            row.sponsors ? embed(typeof row.sponsors === "string" ? row.sponsors : JSON.stringify(row.sponsors), apiKey) : Promise.resolve(null),
            row.family_program_types ? embed(row.family_program_types.join(" "), apiKey) : Promise.resolve(null),
            row.community_programs ? embed(row.community_programs.join(" "), apiKey) : Promise.resolve(null),
            row.cause_partnerships ? embed(row.cause_partnerships.join(" "), apiKey) : Promise.resolve(null),
        ]);

        // false if false or null
        const stadium = row.owns_stadium === true;
        
        // normalize all of the numerical values
        const attendance_norm = (row.avg_game_attendance != null) ? (row.avg_game_attendance - attendance.mean) / attendance.sd : null
        const valuation_norm = (row.franchise_value != null) ? (row.franchise_value - valuation.mean) / valuation.sd : null
        const gdp_norm = (row.metro_gdp != null) ? (row.metro_gdp - gdp.mean) / gdp.sd : null
        const ticket_price_norm = (row.avg_ticket_price != null) ? (row.avg_ticket_price - ticketStats.mean) / ticketStats.sd : null
        const family_programs_norm = (row.family_program_count != null) ? (row.family_program_count - family_programs.mean) / family_programs.sd : null
        const revenue_norm = (row.annual_revenue != null) ? (row.annual_revenue - revenue.mean) / revenue.sd : null
        const population_norm = (row.city_population != null) ? (row.city_population - population.mean) / population.sd : null

        let x_norm = -1;

        if (row.followers_x != null) {
          if (row.followers_x < x.mean) {
            // Range: [0 to mean] maps to [-1 to 0]
            // Result is -1 if followers are 0, and 0 if followers equal the mean
            x_norm = (row.followers_x / x.mean) - 1;
          } else {
            // Range: [mean to max] maps to [0 to 1]
            // Result is 0 if followers equal the mean, and 1 if followers equal max
            // Note: You will need to calculate x.max beforehand
            x_norm = (row.followers_x - x.mean) / (x.max - x.mean);
          }
        }

        let instagram_norm = -1;
        if (row.followers_instagram != null) {
          if (row.followers_instagram < instagram.mean) {
            // Range: [0 to mean] maps to [-1 to 0]
            // Result is -1 if followers are 0, and 0 if followers equal the mean
            instagram_norm = (row.followers_instagram / instagram.mean) - 1;
          } else {
            // Range: [mean to max] maps to [0 to 1]
            // Result is 0 if followers equal the mean, and 1 if followers equal max
            // Note: You will need to calculate instagram.max beforehand
            instagram_norm = (row.followers_instagram - instagram.mean) / (instagram.max - instagram.mean);
          }
        }

        let facebook_norm = -1;
        if (row.followers_facebook != null) {
          if (row.followers_facebook < facebook.mean) {
            // Range: [0 to mean] maps to [-1 to 0]
            // Result is -1 if followers are 0, and 0 if followers equal the mean
            facebook_norm = (row.followers_facebook / facebook.mean) - 1;
          } else {
            // Range: [mean to max] maps to [0 to 1]
            // Result is 0 if followers equal the mean, and 1 if followers equal max
            // Note: You will need to calculate facebook.max beforehand
            facebook_norm = (row.followers_facebook - facebook.mean) / (facebook.max - facebook.mean);
          }
        }

        let tiktok_norm = -1;
        if (row.followers_tiktok != null) {
          if (row.followers_tiktok < tiktok.mean) {
            // Range: [0 to mean] maps to [-1 to 0]
            // Result is -1 if followers are 0, and 0 if followers equal the mean
            tiktok_norm = (row.followers_tiktok / tiktok.mean) - 1;
          } else {
            // Range: [mean to max] maps to [0 to 1]
            // Result is 0 if followers equal the mean, and 1 if followers equal max
            // Note: You will need to calculate tiktok.max beforehand
            tiktok_norm = (row.followers_tiktok - tiktok.mean) / (tiktok.max - tiktok.mean);
          }
        }

        let youtube_norm = -1;
        if (row.subscribers_youtube != null) {
          if (row.subscribers_youtube < youtube.mean) {
            // Range: [0 to mean] maps to [-1 to 0]
            // Result is -1 if followers are 0, and 0 if followers equal the mean
            youtube_norm = (row.subscribers_youtube / youtube.mean) - 1;
          } else {
            // Range: [mean to max] maps to [0 to 1]
            // Result is 0 if followers equal the mean, and 1 if followers equal max
            // Note: You will need to calculate youtube.max beforehand
            youtube_norm = (row.subscribers_youtube - youtube.mean) / (youtube.max - youtube.mean);
          }
        }
        
        const digital_reach_score = (instagram_norm + x_norm + facebook_norm + tiktok_norm + youtube_norm) / 5
        const local_reach_score = ((attendance_norm ?? -1) + (population_norm ?? -1)) / 2

        // Calculate demographic weights
        // YUBI: should I use womenWeight and menWeight?
        const womenWeight = (x_norm != null) ? 0.33*x_norm : null
        const menWeight = (x_norm != null) ? 0.67*x_norm : null


        const genZWeight = (0.5*instagram_norm + 0.5*tiktok_norm)
        const millenialWeight = 0.2*instagram_norm + 0.2*tiktok_norm + 0.2*x_norm + 0.2*facebook_norm + 0.2*youtube_norm
        const genXWeight = 0.33*x_norm + 0.33*facebook_norm + 0.33*youtube_norm
        const boomerWeight = facebook_norm
        const kidsWeight = youtube_norm


        let value_tier_score = 1

        // NOTE: cost tier, 5M = 3, 1M = 2, less = 1
        if (row.franchise_value != null) {
          if (row.franchise_value > 5000) { // in thousands
            value_tier_score = 3
          } else if (row.franchise_value > 1000) { // in thousands
            value_tier_score = 2
          }
        } else if (row.avg_ticket_price != null) {
          if (row.avg_ticket_price > 70) {
            value_tier_score = 3
          } else if (row.avg_ticket_price > 30) {
            value_tier_score = 2
          }
        }

        const cleanRow = {
          name: row.name,
          region: row.region,
          league: row.league,
          category: row.category,
          official_url: row.official_url,
  
          region_embedding: regionEmb,
          league_embedding: leagueEmb,
          values_embedding: valuesEmb,
          sponsors_embedding: sponsorsEmb,
          family_programs_embedding: familyProgramsEmb,
          community_programs_embedding: communityProgramsEmb,
          partners_embedding: partnersEmb,

            digital_reach: digital_reach_score,
            local_reach: local_reach_score,
            family_friendly: family_programs_norm,

            value_tier: value_tier_score,

            women_weight: womenWeight,
            men_weight: menWeight,
            gen_z_weight: genZWeight,
            millenial_weight: millenialWeight,
            gen_x_weight: genXWeight,
            boomer_weight: boomerWeight,
            kids_weight: kidsWeight,
            stadium_ownership: stadium
  
        };
  
        await ctx.runMutation(api.dataPreProcessFull.insertCleanRow, { row: cleanRow });
      }
  
      return "All_Teams_Clean built successfully.";
    },
  });