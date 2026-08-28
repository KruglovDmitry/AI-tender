import { useId } from "react";
import { btnSecondaryClass, mutedTextClass } from "../lib/styles";

interface FileInputProps {
  accept?: string;
  multiple?: boolean;
  label?: string;
  emptyHint?: string;
  selectedHint?: string;
  onChange: (files: File[]) => void;
}

export function FileInput({
  accept,
  multiple,
  label = "Выбрать файлы",
  emptyHint = "Файл не выбран",
  selectedHint,
  onChange,
}: FileInputProps) {
  const id = useId();

  return (
    <div className="flex flex-wrap items-center gap-4">
      <input
        id={id}
        type="file"
        accept={accept}
        multiple={multiple}
        className="hidden"
        onChange={(e) => {
          const list = e.target.files ? Array.from(e.target.files) : [];
          onChange(list);
        }}
      />
      <label htmlFor={id} className={`${btnSecondaryClass} mb-0`}>
        {label}
      </label>
      <span className={mutedTextClass}>
        {selectedHint ?? emptyHint}
      </span>
    </div>
  );
}
