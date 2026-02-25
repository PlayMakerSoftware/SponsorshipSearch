import { v } from "convex/values";
import { action } from "./_generated/server";
import { api } from "./_generated/api";
import { AllTeamsClean, AllTeamsCleanWithoutEmbeddings, stripEmbeddings } from "./All_Teams_Clean";

// ----------------------
// Configuration
// ----------------------

// Page size for batch processing (~100 teams = ~4MB per batch, well under 16MB limit)
const BATCH_SIZE = 100;
// Default number of results to return to frontend
const DEFAULT_RESULT_LIMIT = 50;

// ----------------------
// Similarity Computation
// ----------------------

function cosineSimilarity(a: number[] | null, b: number[] | null): number {
  // 1. Check if either vector is null, undefined, or empty
  if (!a || !b || a.length === 0 || b.length === 0) return 0;

  // 2. Ensure vectors are the same length to avoid undefined multiplication
  if (a.length !== b.length) {
    console.warn("Vector length mismatch:", a.length, b.length);
    return 0;
  }
  
  const dot = a.reduce((sum, val, i) => sum + val * b[i], 0);
  const normA = Math.sqrt(a.reduce((s, v) => s + v * v, 0));
  const normB = Math.sqrt(b.reduce((s, v) => s + v * v, 0));
  if (normA === 0 || normB === 0) return 0;

  const similarity = dot / (normA * normB);
  return isNaN(similarity) ? 0 : similarity;
}

// ----------------------
// Embedding Helper
// ----------------------

// YUBI: added embedding logic, make sure that this is correct
async function embedText(txt: string | undefined | null): Promise<number[] | null> {
    if (!txt || txt.trim() === "") return null;

    // 1. Access the environment variable via process.env
    const apiKey = process.env.GEMINI_API_KEY;
  
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

// ----------------------
// Types
// ----------------------

type BrandVector = {
  region_embedding: number[] | null;
  league_embedding: number[] | null;
  values_embedding: number[] | null;
  audience_embedding: number[] | null;
  goals_embedding: number[] | null;
  query_embedding: number[] | null;
};

type ScoringContext = {
  brandVector: BrandVector;
  brandLeagues: string[]; // Array of selected league filter values
  brandRegion: string;
  brandAudience: string;
  brandGoals: string;
  target_value_tier: number;
};

// ----------------------
// Sport/League Matching
// ----------------------

/**
 * Maps filter values to patterns that match actual league names in the database
 * Filter values: "NFL", "NBA G League WNBA", "MLB MiLB", "NHL ECHL AHL", "MLS NWSL"
 * Actual leagues: "National Football League", "Major League Baseball — American League", etc.
 */
function teamMatchesSportFilter(teamLeague: string, filterValues: string[]): boolean {
  if (filterValues.length === 0) return true; // No filter = match all
  
  const leagueLower = teamLeague.toLowerCase();
  
  for (const filter of filterValues) {
    // NFL / Football
    if (filter.includes("NFL") && (
      leagueLower.includes("national football league") ||
      leagueLower.includes("nfl")
    )) {
      return true;
    }
    
    // Basketball: NBA, G League, WNBA
    if (filter.includes("NBA") && (
      leagueLower.includes("national basketball association") ||
      leagueLower.includes("nba") ||
      leagueLower.includes("g league") ||
      leagueLower.includes("women's national basketball")
    )) {
      return true;
    }
    
    // Baseball: MLB, MiLB (minor leagues)
    if (filter.includes("MLB") && (
      leagueLower.includes("major league baseball") ||
      leagueLower.includes("triple-a") ||
      leagueLower.includes("double-a") ||
      leagueLower.includes("single-a") ||
      leagueLower.includes("high-a") ||
      leagueLower.includes("class a") ||
      leagueLower.includes("rookie") ||
      leagueLower.includes("milb")
    )) {
      return true;
    }
    
    // Hockey: NHL, AHL, ECHL
    if (filter.includes("NHL") && (
      leagueLower.includes("national hockey league") ||
      leagueLower.includes("nhl") ||
      leagueLower.includes("american hockey league") ||
      leagueLower.includes("ahl") ||
      leagueLower.includes("echl")
    )) {
      return true;
    }
    
    // Soccer: MLS, NWSL
    if (filter.includes("MLS") && (
      leagueLower.includes("major league soccer") ||
      leagueLower.includes("mls") ||
      leagueLower.includes("national women's soccer") ||
      leagueLower.includes("nwsl")
    )) {
      return true;
    }
  }
  
  return false;
}

export type ScoredTeamWithoutEmbeddings = AllTeamsCleanWithoutEmbeddings & { 
  similarity_score: number;
};

// ----------------------
// Scoring Logic (extracted for reuse)
// ----------------------

function computeTeamScore(team: AllTeamsClean, ctx: ScoringContext): number {
  const { brandVector, brandLeagues, brandRegion, brandAudience, brandGoals, target_value_tier } = ctx;

  // FIRST: Check if team matches the sport filter - if not, return 0 immediately
  // This ensures teams from non-selected sports are excluded from results
  if (brandLeagues.length > 0 && !teamMatchesSportFilter(team.league, brandLeagues)) {
    return 0;
  }

  // scale is close to 0.7 to 0.9
  const simRegion = Math.max(0, cosineSimilarity(brandVector.region_embedding, team.region_embedding));

  // filter out teams that don't match region specified by brand
  // only check if 1 region is selected
  if (brandRegion.length > 1 && brandRegion.length < 25) {
    if (simRegion < 0.6) return 0; // loosen similarity
  }

  // scale is close to 0.7 to 0.9
  const simValues = Math.max(0, cosineSimilarity(brandVector.values_embedding, team.values_embedding));

  // aggregate value and audience and query together
  let simQuery = Math.max(0, cosineSimilarity(brandVector.query_embedding, team.league_embedding));
  simQuery += Math.max(0, cosineSimilarity(brandVector.query_embedding, team.values_embedding));
  simQuery += Math.max(0, cosineSimilarity(brandVector.query_embedding, team.community_programs_embedding));
  simQuery /= 3;

  // YUBI: this has a range of 0 to 1
  const tierDiff = Math.abs(target_value_tier - (team.value_tier ?? 1));
  const valuationSim = Math.max(0, 1 - (tierDiff / 2)); // 0 diff = 1.0 score; 2 diff = 0.0 score

  // Set target value tier of team using goals
  let demSim = 0;
  let demCounter = 0;
  const leagueLower = team.league.toLowerCase();
  
  // adjust so floor is 0
  if (brandAudience.includes("gen-z")) {
    demSim += (team.gen_z_weight ?? -1) + 1;
    demCounter += 1;
  }
  if (brandAudience.includes("millennials")) {
    demSim += (team.millenial_weight ?? -1) + 1;
    demCounter += 1;
  } 
  if (brandAudience.includes("gen-x")) {
    demSim += (team.gen_x_weight ?? -1) + 1;
    demCounter += 1;
  } 
  if (brandAudience.includes("boomer")) {
    demSim += (team.boomer_weight ?? -1) + 1;
    demCounter += 1;
  } 
  if (brandAudience.includes("kids")) {
    demSim += (team.kids_weight ?? -1) + 1;
    demCounter += 1;
  } 
  if (brandAudience.includes("women")) {
    demSim += (team.women_weight ?? 0) + 1;
    // Boost for women's leagues
    if (leagueLower.includes("women's national basketball") || leagueLower.includes("national women's soccer")) {
      demSim += 1;
    }
    demCounter += 1;
  } 
  if (brandAudience.includes("families")) {
    demSim += team.family_friendly ?? 0;
    demCounter += 1;
  }

  // Normalize demSim
  // YUBI: set to 0.2 so that demSim is never 0
  demSim = demCounter > 0 ? Math.min(demSim / demCounter, 1) : 0.2;
  demSim = Math.max(0.2, demSim);

  // set reach score
  // adjust so floor is 0
  let reachSim = 0;
  if (brandGoals.includes("digital-presence")) {
    reachSim = (team.digital_reach ?? -1) + 1;
  } else if (brandGoals.includes("local-presence")) {
    reachSim = (team.local_reach ?? -1) + 1;
  } else {
    reachSim = (((team.digital_reach ?? -1) + 1) + ((team.local_reach ?? -1) + 1)) / 2;
  }
  reachSim = Math.max(0, reachSim);

  // YUBI: modify weights as desired
  const WEIGHTS = {
    region: 0.5,
    query: 0.03,
    values: 0.04,
    valuation: 0.2,
    demographics: 0.2,
    reach: 0.03
  };

  // We multiply each score by its weight
  let weightedScore =
    (simRegion * WEIGHTS.region) +
    (simQuery * WEIGHTS.query) +
    (simValues * WEIGHTS.values) +
    (valuationSim * WEIGHTS.valuation) +
    (demSim * WEIGHTS.demographics) +
    (reachSim * WEIGHTS.reach);

  if (weightedScore > 1) {
    weightedScore = 1;
  } else if (weightedScore <= 0) {
    // YUBI: prevent score from being less than 0
    weightedScore = 0.1;
  }

  return weightedScore;
}

/**
 * Insert a scored team into a sorted array, maintaining only top N results
 * Uses binary search for efficient insertion
 */
function insertIntoTopN(
  topResults: ScoredTeamWithoutEmbeddings[],
  newTeam: ScoredTeamWithoutEmbeddings,
  maxSize: number
): void {
  const score = newTeam.similarity_score;
  
  // Skip if we're full and this score is worse than the worst in our list
  if (topResults.length >= maxSize && score <= topResults[topResults.length - 1].similarity_score) {
    return;
  }

  // Binary search for insertion position (descending order)
  let left = 0;
  let right = topResults.length;
  while (left < right) {
    const mid = Math.floor((left + right) / 2);
    if (topResults[mid].similarity_score > score) {
      left = mid + 1;
    } else {
      right = mid;
    }
  }

  // Insert at the found position
  topResults.splice(left, 0, newTeam);

  // Remove the last element if we exceed maxSize
  if (topResults.length > maxSize) {
    topResults.pop();
  }
}

// ----------------------
// Response Types
// ----------------------

export type PaginatedSimilarityResponse = {
  teams: ScoredTeamWithoutEmbeddings[];
  totalCount: number;
  totalPages: number;
  currentPage: number;
  pageSize: number;
  hasNextPage: boolean;
  hasPreviousPage: boolean;
};

// ----------------------
// Convex Action: Compute Similarity (with pagination)
// ----------------------

export const computeBrandSimilarity = action({
  args: {
    query: v.string(),
    filters: v.object({
      regions: v.array(v.string()),
      demographics: v.array(v.string()),
      brandValues: v.array(v.string()),
      leagues: v.array(v.string()),
      goals: v.array(v.string()),
      budgetMin: v.optional(v.number()),
      budgetMax: v.optional(v.number()),
    }),
    page: v.optional(v.number()),     // Page number (1-indexed, default: 1)
    pageSize: v.optional(v.number()), // Results per page (default: 20)
  },

  handler: async (ctx, args): Promise<PaginatedSimilarityResponse> => {
    const { query, filters, page = 1, pageSize = 20 } = args;

    // ------------------------------------------------------------
    // 1. Build Brand Vector (Embeddings + Normalized Numeric Inputs)
    // ------------------------------------------------------------

    const brandRegion = filters.regions.join(" ");
    const brandLeagueText = filters.leagues.join(" "); // For embedding
    const brandLeagues = filters.leagues; // Array for filtering
    const brandValues = filters.brandValues.join(" ");
    const brandAudience = filters.demographics.join(" ");
    const brandGoals = filters.goals.join(" ");

    // Set target value tier of team using goals
    let target_value_tier = 2;
    if (brandGoals.includes("prestige-credibility")) {
      target_value_tier += 1;
    } else if (brandGoals.includes("brand-awareness")) {
      target_value_tier += 1;
    } else if (brandGoals.includes("business-to-business")) {
      target_value_tier += 1;
    } else if (brandGoals.includes("fan-connection-activation-control")) {
      target_value_tier -= 2;
    }

    // constrain the target value tier to be within 1 and 3
    target_value_tier = Math.max(1, Math.min(3, target_value_tier));

    const brandVector: BrandVector = {
      region_embedding: await embedText(brandRegion),
      league_embedding: await embedText(brandLeagueText),
      values_embedding: await embedText(brandValues),
      audience_embedding: await embedText(brandAudience),
      goals_embedding: await embedText(brandGoals),
      query_embedding: await embedText(query)
    };

    const scoringContext: ScoringContext = {
      brandVector,
      brandLeagues, // Now an array for proper filtering
      brandRegion, // still a string
      brandAudience,
      brandGoals,
      target_value_tier,
    };

    // ------------------------------------------------------------
    // 2. Process Teams in Batches (to stay within Convex limits)
    // ------------------------------------------------------------

    const allScoredTeams: ScoredTeamWithoutEmbeddings[] = [];
    let cursor: string | undefined = undefined;
    let hasMore = true;

    while (hasMore) {
      // Fetch a page of teams with embeddings
      const dbPage: { teams: AllTeamsClean[]; nextCursor: string | null; isDone: boolean } = 
        await ctx.runQuery(api.All_Teams_Clean.getPage, {
          cursor,
          limit: BATCH_SIZE,
        });

      // Process each team in the batch
      for (const team of dbPage.teams) {
        const score = computeTeamScore(team, scoringContext);
        
        // Strip embeddings and add score
        const scoredTeam: ScoredTeamWithoutEmbeddings = {
          ...stripEmbeddings(team),
          similarity_score: score,
        };

        allScoredTeams.push(scoredTeam);
      }

      // Check if there are more pages
      hasMore = !dbPage.isDone;
      cursor = dbPage.nextCursor ?? undefined;
    }

    // ------------------------------------------------------------
    // 3. Filter out teams with score 0 (filtered out by sport)
    //    and sort by similarity score (descending)
    // ------------------------------------------------------------

    const filteredTeams = allScoredTeams.filter(t => t.similarity_score > 0);
    filteredTeams.sort((a, b) => b.similarity_score - a.similarity_score);

    // ------------------------------------------------------------
    // 4. Calculate pagination and return requested page
    // ------------------------------------------------------------

    const totalCount = filteredTeams.length;
    const totalPages = Math.ceil(totalCount / pageSize);
    const currentPage = Math.max(1, Math.min(page, totalPages || 1));
    
    const startIndex = (currentPage - 1) * pageSize;
    const endIndex = Math.min(startIndex + pageSize, totalCount);
    
    const paginatedTeams = filteredTeams.slice(startIndex, endIndex);

    return {
      teams: paginatedTeams,
      totalCount,
      totalPages,
      currentPage,
      pageSize,
      hasNextPage: currentPage < totalPages,
      hasPreviousPage: currentPage > 1,
    };
  },
});