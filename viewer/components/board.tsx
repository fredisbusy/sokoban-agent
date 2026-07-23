import type { CSSProperties } from "react";

import { parseBoard } from "../lib/board";

interface BoardProps {
  board: string | null;
}

type PieceStyle = CSSProperties & {
  "--row": number;
  "--column": number;
};

export function Board({ board }: BoardProps) {
  if (!board) {
    return (
      <div className="empty-board">
        <span className="empty-mark">@</span>
        <p>실행하면 실제 graph state가 여기에 표시됩니다.</p>
      </div>
    );
  }

  const rows = parseBoard(board);
  const tiles = rows.flat();
  const style = {
    "--rows": rows.length,
    "--columns": rows[0].length,
  } as CSSProperties;

  return (
    <div
      className="board"
      style={style}
      role="img"
      aria-label={`Sokoban 보드, ${rows.length}행 ${rows[0].length}열`}
    >
      {tiles.map((tile) => (
        <span
          className={`tile${tile.wall ? " wall" : ""}${tile.target ? " target" : ""}`}
          aria-hidden="true"
          key={`tile-${tile.row}-${tile.column}`}
        />
      ))}
      {tiles.filter((tile) => tile.box).map((tile, index) => (
        <span
          className={`piece box${tile.target ? " complete" : ""}`}
          style={{ "--row": tile.row, "--column": tile.column } as PieceStyle}
          aria-label="상자"
          key={`box-${index}`}
        />
      ))}
      {tiles.filter((tile) => tile.player).map((tile) => (
        <span
          className="piece player"
          style={{ "--row": tile.row, "--column": tile.column } as PieceStyle}
          aria-label="플레이어"
          key="player"
        />
      ))}
    </div>
  );
}
