import type { BoardTile } from "./types";

const SYMBOLS = new Set(["#", " ", ".", "$", "@", "*", "+"]);
const OBSERVATION_SYMBOLS = [" ", "#", ".", "$", "@", "*", "+"];

export function parseBoard(board: string): BoardTile[][] {
  if (typeof board !== "string" || board.length === 0) {
    throw new TypeError("board must be a non-empty string");
  }
  const rows = board.replace(/\r/g, "").split("\n");
  if (rows.at(-1) === "") rows.pop();
  const columns = Math.max(...rows.map((row) => row.length));
  if (columns === 0) throw new TypeError("board must contain tiles");

  return rows.map((row, rowIndex) =>
    row.padEnd(columns, " ").split("").map((symbol, columnIndex) => {
      if (!SYMBOLS.has(symbol)) {
        throw new TypeError(`unknown board symbol: ${symbol}`);
      }
      return {
        symbol,
        row: rowIndex,
        column: columnIndex,
        wall: symbol === "#",
        target: ".+*".includes(symbol),
        box: "$*".includes(symbol),
        player: "@+".includes(symbol),
      };
    }),
  );
}

export function boardFromObservation(observation: unknown): string | null {
  if (!Array.isArray(observation) || observation.length === 0) return null;
  const rows = observation.map((row) => {
    if (!Array.isArray(row)) throw new TypeError("observation rows must be arrays");
    return row.map((value) => {
      if (!Number.isInteger(value)) throw new TypeError(`unknown observation tile: ${value}`);
      const symbol = OBSERVATION_SYMBOLS[value as number];
      if (symbol === undefined) throw new TypeError(`unknown observation tile: ${value}`);
      return symbol;
    }).join("");
  });
  return rows.join("\n");
}
