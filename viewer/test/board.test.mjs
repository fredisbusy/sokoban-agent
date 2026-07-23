import assert from "node:assert/strict";
import test from "node:test";

import { boardFromObservation, parseBoard } from "../src/board.js";

test("numeric observation becomes Sokoban symbols", () => {
  assert.equal(
    boardFromObservation([[1, 1, 1], [1, 4, 1], [2, 5, 6]]),
    "###\n#@#\n.*+",
  );
});

test("board parser exposes semantic layers and pads ragged floor", () => {
  const board = parseBoard("###\n#@ \n#*");
  assert.equal(board.length, 3);
  assert.equal(board[0].length, 3);
  assert.equal(board[1][1].player, true);
  assert.equal(board[2][1].box, true);
  assert.equal(board[2][1].target, true);
  assert.equal(board[2][2].symbol, " ");
});

test("board parser rejects unknown symbols", () => {
  assert.throws(() => parseBoard("#x#"), /unknown board symbol/);
});
