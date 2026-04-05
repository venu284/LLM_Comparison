export enum Status {
  Pending = 'Pending',
  Active = 'Active',
  Inactive = 'Inactive',
  Archived = 'Archived'
}

export type StatusFilter = Status | Status[];

export interface StatusItem {
  id: string;
  status: Status;
}

export function filterByStatus(items: StatusItem[], filter: StatusFilter): StatusItem[] {
  const allowedStatuses = Array.isArray(filter) ? filter : [filter];
  return items.filter((item) => allowedStatuses.includes(item.status));
}

